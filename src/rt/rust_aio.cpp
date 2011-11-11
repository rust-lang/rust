// Disable libev prototypes - they will make inline compatibility functions
// which are unused and so trigger a warning in gcc since -Wall is on.
#define EV_PROTOTYPES 0
#include "uv.h"

#include "rust_internal.h"
#include "rust_scheduler.h"
#include "rust_upcall.h"

#ifdef __GNUC__
#define LOG_CALLBACK_ENTRY(p) \
    LOG(iotask, callback, "> IO CALLBACK %s %p", __FUNCTION__, p)
#else
#define LOG_CALLBACK_ENTRY(p) \
    LOG(iotask, callback, "> IO CALLBACK %s:%d %p", __FILE__, __LINE__, p)
#endif

// The primary task which is running the event loop. This is used to dispatch
// all the notifications back to rust so we clone all passed in channels to
// this task.
static rust_task *iotask = NULL;

struct socket_data : public task_owned<socket_data> {
  // Either the task that the connection attempt was made from or the task
  // that the server was spawned on.
  rust_task *task;
  // Channel for reporting the status of a connection attempt
  // For connections from servers, this is always null
  // For server sockets, this is used to send the notification that the server
  // was closed.
  rust_chan *chan;
  // Channel to a port which receives bytes from this socket
  rust_chan *reader;
  uv_tcp_t socket;

  ~socket_data() {
    if (chan)
      chan->deref();
    if (reader)
      reader->deref();
  }

  void send_result(void *data) {
    chan->port->send(&data);
    chan->deref();
    chan = NULL;
  }
};

struct req_connect : public uv_connect_t, public task_owned<req_connect> {};
struct req_write : public uv_write_t, public task_owned<req_write> {};

extern "C" CDECL void aio_close_socket(socket_data *);

static uv_idle_s idle_handler;

static void idle_callback(uv_idle_t* handle, int status) {
  rust_task *task = reinterpret_cast<rust_task*>(handle->data);
  task->yield();
}

extern "C" CDECL void aio_init() {
  rust_task *task = rust_scheduler::get_task();
  LOG_UPCALL_ENTRY(task);
  iotask = task;
  uv_idle_init(uv_default_loop(), &idle_handler);
  uv_idle_start(&idle_handler, idle_callback);
}

extern "C" CDECL void aio_run() {
  rust_task *task = rust_scheduler::get_task();
  LOG_UPCALL_ENTRY(task);
  idle_handler.data = task;
  uv_run(uv_default_loop());
}

void nop_close(uv_handle_t* handle) {}

extern "C" CDECL void aio_stop() {
  rust_task *task = rust_scheduler::get_task();
  LOG_UPCALL_ENTRY(task);
  uv_close((uv_handle_t*)&idle_handler, nop_close);
}

static socket_data *make_socket(rust_task *task, rust_chan *chan) {
  socket_data *data = new (task, "make_socket") socket_data;
  if (!data ||
      uv_tcp_init(uv_default_loop(), &data->socket)) {
    return NULL;
  }
  data->socket.data = data;
  data->task = task;
  // Connections from servers don't have a channel
  if (chan) {
    data->chan = chan->clone(iotask);
  } else {
    data->chan = NULL;
  }
  data->socket.data = data;
  data->reader = NULL;
  return data;
}

// We allocate the requested space + rust_vec but return a pointer at a
// +rust_vec offset so that it writes the bytes to the correct location.
static uv_buf_t alloc_buffer(uv_handle_t *socket, size_t suggested_size) {
  LOG_CALLBACK_ENTRY(socket);
  uv_buf_t buf;
  size_t actual_size = suggested_size + sizeof (rust_vec);
  socket_data *data = (socket_data*)socket->data;
  char *base =
    reinterpret_cast<char*>(data->task->kernel->malloc(actual_size,
                                                       "read buffer"));
  buf.base = base + sizeof (rust_vec);
  buf.len = suggested_size;
  return buf;
}

static void read_progress(uv_stream_t *socket, ssize_t nread, uv_buf_t buf) {
  LOG_CALLBACK_ENTRY(socket);
  socket_data *data = (socket_data*)socket->data;
  I(data->task->sched, data->reader != NULL);
  I(data->task->sched, nread <= ssize_t(buf.len));

  rust_vec *v = reinterpret_cast<rust_vec*>(
      reinterpret_cast<char*>(buf.base) - sizeof (rust_vec));
  v->alloc = buf.len;

  switch (nread) {
    case -1: // End of stream
      v->fill = 0;
      uv_read_stop(socket);
      break;
    case 0: // Nothing read
      data->task->kernel->free(v);
      return;
    default: // Got nread bytes
      v->fill = nread;
      break;
  }
  data->reader->port->send(v);
}

static void new_connection(uv_stream_t *socket, int status) {
  LOG_CALLBACK_ENTRY(socket);
  socket_data *server = (socket_data*)socket->data;
  I(server->task->sched, (uv_tcp_t*)socket == &server->socket);
  // Connections from servers don't have a channel
  socket_data *client = make_socket(server->task, NULL);
  if (!client) {
    server->task->fail();
    return;
  }
  if (uv_accept(socket, (uv_stream_t*)&client->socket)) {
    aio_close_socket(client);
    server->task->fail();
    return;
  }
  server->chan->port->send(&client);
}

extern "C" CDECL socket_data *aio_serve(const char *ip, int port,
                                        chan_handle *_chan) {
  rust_task *task = rust_scheduler::get_task();
  LOG_UPCALL_ENTRY(task);
  rust_chan *chan = task->get_chan_by_handle(_chan);
  if(!chan) return NULL;
  struct sockaddr_in addr = uv_ip4_addr(const_cast<char*>(ip), port);
  socket_data *server = make_socket(iotask, chan);
  if (!server)
    goto oom;
  if (uv_tcp_bind(&server->socket, addr) ||
      uv_listen((uv_stream_t*)&server->socket, 128, new_connection)) {
    aio_close_socket(server);
    chan->deref();
    return NULL;
  }
  chan->deref();
  return server;
oom:
  chan->deref();
  task->fail();
  return NULL;
}

static void free_socket(uv_handle_t *handle) {
  LOG_CALLBACK_ENTRY(socket);
  uv_tcp_t *socket = (uv_tcp_t*)handle;
  socket_data *data = (socket_data*)socket->data;
  I(data->task->sched, socket == &data->socket);
  // For client sockets, send a 0-size buffer to indicate that we're done
  // reading and should send the close notification.
  if (data->reader) {
    if (data->reader->is_associated()) {
      uv_buf_t buf = alloc_buffer((uv_handle_t*)socket, 0);
      read_progress((uv_stream_t*)socket, -1, buf);
      uv_read_stop((uv_stream_t*)socket);
    }
  } else {
    // This is a server socket
    bool closed = true;
    I(data->task->sched, data->chan != NULL);
    data->task->kill();
    data->chan->port->send(&closed);
  }
  delete data;
}

extern "C" CDECL void aio_close_socket(socket_data *client) {
  rust_task *task = rust_scheduler::get_task();
  LOG_UPCALL_ENTRY(task);
  uv_close((uv_handle_t*)&client->socket, free_socket);
}

extern "C" CDECL void aio_close_server(socket_data *server,
                                       chan_handle *_chan) {
  rust_task *task = rust_scheduler::get_task();
  LOG_UPCALL_ENTRY(task);
  rust_chan *chan = task->get_chan_by_handle(_chan);
  if(!chan) return;
  // XXX: hax until rust_task::kill
  // send null and the receiver knows to call back into native code to check
  void* null_client = NULL;
  server->chan->port->send(&null_client);
  server->chan->deref();
  server->chan = chan->clone(iotask);
  aio_close_socket(server);
  chan->deref();
}

extern "C" CDECL bool aio_is_null_client(socket_data *server) {
  rust_task *task = rust_scheduler::get_task();
  LOG_UPCALL_ENTRY(task);
  return server == NULL;
}

static void connection_complete(uv_connect_t *req, int status) {
  LOG_CALLBACK_ENTRY(socket);
  socket_data *client = (socket_data*)req->data;
  client->send_result(client);
  free(req);
}

extern "C" CDECL void aio_connect(const char *host, int port,
                                  chan_handle *_chan) {
  rust_task *task = rust_scheduler::get_task();
  LOG_UPCALL_ENTRY(task);
  rust_chan *chan = task->get_chan_by_handle(_chan);
  uv_connect_t *req = NULL;
  if(!chan) return;
  struct sockaddr_in addr = uv_ip4_addr(const_cast<char*>(host), port);
  socket_data *client = make_socket(iotask, NULL);
  if (!client) {
    goto oom_client;
  }
  req = (uv_connect_t*)client->task->malloc(
      sizeof(uv_connect_t), "connection request");
  if (!req) {
    goto oom_req;
  }
  req->data = client;
  if (0 == uv_tcp_connect(req, &client->socket, addr, connection_complete)) {
      chan->deref();
      return;
  }
  free(req);
oom_req:
  aio_close_socket(client);
oom_client:
  chan->deref();
  task->fail();
  return;
}

static void write_complete(uv_write_t *req, int status) {
  LOG_CALLBACK_ENTRY(socket);
  bool success = status == 0;
  socket_data *client = (socket_data*)req->data;
  client->send_result(&success);
  free(req);
}

extern "C" CDECL void aio_writedata(socket_data *data, char *buf,
                                    size_t size, chan_handle *_chan) {
  rust_task *task = rust_scheduler::get_task();
  LOG_UPCALL_ENTRY(task);
  rust_chan *chan = task->get_chan_by_handle(_chan);
  uv_write_t *req;
  if(!chan) return;

  // uv_buf_t is defined backwards on win32...
  // maybe an indication we shouldn't be building directly?
#if defined(__WIN32__)
  uv_buf_t buffer = { size, buf };
#else
  uv_buf_t buffer = { buf, size };
#endif

  req = (uv_write_t*)data->task->malloc(sizeof(uv_write_t), "write request");
  if (!req) {
    goto fail;
  }
  req->data = data;
  if (uv_write(req, (uv_stream_t*)&data->socket, &buffer, 1,
               write_complete)) {
    free(req);
    goto fail;
  }
  chan->deref();
  return;
fail:
  chan->deref();
  task->fail();
}

extern "C" CDECL void aio_read(socket_data *data, chan_handle *_chan) {
  rust_task *task = rust_scheduler::get_task();
  LOG_UPCALL_ENTRY(task);
  rust_chan *reader = task->get_chan_by_handle(_chan);
  if(!reader) return;
  I(task->sched, data->reader == NULL);
  data->reader = reader->clone(iotask);
  uv_read_start((uv_stream_t*)&data->socket, alloc_buffer, read_progress);
  reader->deref();
}

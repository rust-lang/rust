#include "rust_internal.h"
#include "rust_upcall.h"
// Disable libev prototypes - they will make inline compatability functions
// which are unused and so trigger a warning in gcc since -Wall is on.
#define EV_PROTOTYPES 0
#include "uv.h"

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
};

struct request : public uv_req_t, public task_owned<request> {
  rust_task *task;
  // Used for notifying about completion of connections, writes
  rust_chan *chan;
  request(socket_data *data, rust_chan *chan,
          void (*cb)(request *req, int status)) {
    uv_req_init(this, (uv_handle_t*)&data->socket, (void*(*)(void*))cb);
    this->data = data;
    this->task = data->task;
    this->chan = chan->clone(iotask);
  }
  socket_data *socket() {
    return (socket_data*)data;
  }
  void send_result(void *data) {
    chan->send(&data);
    chan->deref();
    chan = NULL;
  }
};

extern "C" CDECL void aio_close_socket(rust_task *task, socket_data *);

static uv_idle_s idle_handler;

static void idle_callback(uv_idle_t* handle, int status) {
  rust_task *task = reinterpret_cast<rust_task*>(handle->data);
  task->yield();
}

extern "C" CDECL void aio_init(rust_task *task) {
  LOG_UPCALL_ENTRY(task);
  iotask = task;
  uv_init();
  uv_idle_init(&idle_handler);
  uv_idle_start(&idle_handler, idle_callback);
}

extern "C" CDECL void aio_run(rust_task *task) {
  LOG_UPCALL_ENTRY(task);
  idle_handler.data = task;
  uv_run();
}

void nop_close(uv_handle_t* handle) {}

extern "C" CDECL void aio_stop(rust_task *task) {
  LOG_UPCALL_ENTRY(task);
  uv_close((uv_handle_t*)&idle_handler, nop_close);
}

static socket_data *make_socket(rust_task *task, rust_chan *chan) {
  socket_data *data = new (task, "make_socket") socket_data;
  if (!data ||
      uv_tcp_init(&data->socket)) {
    return NULL;
  }
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
static uv_buf_t alloc_buffer(uv_stream_t *socket, size_t suggested_size) {
  LOG_CALLBACK_ENTRY(socket);
  uv_buf_t buf;
  size_t actual_size = suggested_size + sizeof (rust_ivec_heap);
  socket_data *data = (socket_data*)socket->data;
  char *base =
    reinterpret_cast<char*>(data->task->kernel->malloc(actual_size,
                                                       "read buffer"));
  buf.base = base + sizeof (rust_ivec_heap);
  buf.len = suggested_size;
  return buf;
}

static void read_progress(uv_stream_t *socket, ssize_t nread, uv_buf_t buf) {
  LOG_CALLBACK_ENTRY(socket);
  socket_data *data = (socket_data*)socket->data;
  I(data->task->sched, data->reader != NULL);
  I(data->task->sched, nread <= ssize_t(buf.len));

  rust_ivec_heap *base = reinterpret_cast<rust_ivec_heap*>(
      reinterpret_cast<char*>(buf.base) - sizeof (rust_ivec_heap));
  rust_ivec v;
  v.fill = 0;
  v.alloc = buf.len;
  v.payload.ptr = base;

  switch (nread) {
    case -1: // End of stream
      base->fill = 0;
      uv_read_stop(socket);
      break;
    case 0: // Nothing read
      data->task->kernel->free(base);
      return;
    default: // Got nread bytes
      base->fill = nread;
      break;
  }
  data->reader->send(&v);
}

static void new_connection(uv_handle_t *socket, int status) {
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
    aio_close_socket(client->task, client);
    server->task->fail();
    return;
  }
  server->chan->send(&client);
}

extern "C" CDECL socket_data *aio_serve(rust_task *task, const char *ip,
                                        int port, chan_handle *_chan) {
  LOG_UPCALL_ENTRY(task);
  rust_chan *chan = task->get_chan_by_handle(_chan);
  if(!chan) return NULL;
  struct sockaddr_in addr = uv_ip4_addr(const_cast<char*>(ip), port);
  socket_data *server = make_socket(iotask, chan);
  if (!server)
    goto oom;
  if (uv_tcp_bind(&server->socket, addr) ||
      uv_tcp_listen(&server->socket, 128, new_connection)) {
    aio_close_socket(task, server);
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
      uv_buf_t buf = alloc_buffer((uv_stream_t*)socket, 0);
      read_progress((uv_stream_t*)socket, -1, buf);
      uv_read_stop((uv_stream_t*)socket);
    }
  } else {
    // This is a server socket
    bool closed = true;
    I(data->task->sched, data->chan != NULL);
    data->task->kill();
    data->chan->send(&closed);
  }
  delete data;
}

extern "C" CDECL void aio_close_socket(rust_task *task, socket_data *client) {
  LOG_UPCALL_ENTRY(task);
  if (uv_close((uv_handle_t*)&client->socket, free_socket)) {
    task->fail();
  }
}

extern "C" CDECL void aio_close_server(rust_task *task, socket_data *server,
                                       chan_handle *_chan) {
  LOG_UPCALL_ENTRY(task);
  rust_chan *chan = task->get_chan_by_handle(_chan);
  if(!chan) return;
  // XXX: hax until rust_task::kill
  // send null and the receiver knows to call back into native code to check
  void* null_client = NULL;
  server->chan->send(&null_client);
  server->chan->deref();
  server->chan = chan->clone(iotask);
  aio_close_socket(task, server);
  chan->deref();
}

extern "C" CDECL bool aio_is_null_client(rust_task *task,
                                         socket_data *server) {
  LOG_UPCALL_ENTRY(task);
  return server == NULL;
}

static void connection_complete(request *req, int status) {
  LOG_CALLBACK_ENTRY(socket);
  socket_data *client = req->socket();
  req->send_result(client);
  delete req;
}

extern "C" CDECL void aio_connect(rust_task *task, const char *host,
                                  int port, chan_handle *_chan) {
  LOG_UPCALL_ENTRY(task);
  rust_chan *chan = task->get_chan_by_handle(_chan);
  if(!chan) return;
  struct sockaddr_in addr = uv_ip4_addr(const_cast<char*>(host), port);
  request *req;
  socket_data *client = make_socket(iotask, NULL);
  if (!client) {
    goto oom_client;
  }
  req = new (client->task, "connection request")
    request(client, chan, connection_complete);
  if (!req) {
    goto oom_req;
  }
  if (0 == uv_tcp_connect(req, addr)) {
      chan->deref();
      return;
  }
oom_req:
  aio_close_socket(task, client);
oom_client:
  chan->deref();
  task->fail();
  return;
}

static void write_complete(request *req, int status) {
  LOG_CALLBACK_ENTRY(socket);
  bool success = status == 0;
  req->send_result(&success);
  delete req;
}

extern "C" CDECL void aio_writedata(rust_task *task, socket_data *data,
                                    char *buf, size_t size,
                                    chan_handle *_chan) {
  LOG_UPCALL_ENTRY(task);
  rust_chan *chan = task->get_chan_by_handle(_chan);
  if(!chan) return;

  // uv_buf_t is defined backwards on win32...
  // maybe an indication we shouldn't be building directly?
#if defined(__WIN32__)
  uv_buf_t buffer = { size, buf };
#else
  uv_buf_t buffer = { buf, size };
#endif

  request *req = new (data->task, "write request")
    request(data, chan, write_complete);
  if (!req) {
    goto fail;
  }
  if (uv_write(req, &buffer, 1)) {
    delete req;
    goto fail;
  }
  chan->deref();
  return;
fail:
  chan->deref();
  task->fail();
}

extern "C" CDECL void aio_read(rust_task *task, socket_data *data,
                               chan_handle *_chan) {
  LOG_UPCALL_ENTRY(task);
  rust_chan *reader = task->get_chan_by_handle(_chan);
  if(!reader) return;
  I(task->sched, data->reader == NULL);
  data->reader = reader->clone(iotask);
  uv_read_start((uv_stream_t*)&data->socket, alloc_buffer, read_progress);
  reader->deref();
}

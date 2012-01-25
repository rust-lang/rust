#include <utility>
#include <queue>
#include <string>
#include "rust_internal.h"
#include "uv.h"

class rust_uvtmp_thread;

struct connect_data {
    uint32_t req_id;
    rust_uvtmp_thread *thread;
    char * ip_addr;
    uv_connect_t connect;
    uv_tcp_t tcp;
    chan_handle chan;
};

const intptr_t connected_tag = 1;
const intptr_t wrote_tag = 2;
const intptr_t read_tag = 3;

struct iomsg {
    intptr_t tag;
    union {
	connect_data *connected_val;
	connect_data *wrote_val;
	struct {
	    connect_data *cd;
	    uint8_t *buf;
	    ssize_t nread;
	} read_val;
    } val;
};

struct write_data {
    connect_data *cd;
    uint8_t *buf;
    size_t len;
    chan_handle chan;
};

struct read_start_data {
    connect_data *cd;
    chan_handle chan;
};

// FIXME: Copied from rust_builtins.cpp. Could bitrot easily
static void
send(rust_task *task, chan_handle chan, void *data) {
    rust_task *target_task = task->kernel->get_task_by_id(chan.task);
    if(target_task) {
        rust_port *port = target_task->get_port_by_id(chan.port);
        if(port) {
            port->send(data);
            scoped_lock with(target_task->lock);
            port->deref();
        }
        target_task->deref();
    }
}

class rust_uvtmp_thread : public rust_thread {

private:
    std::map<int, connect_data *> req_map;
    rust_task *task;
    uv_loop_t *loop;
    uv_idle_t idle;
    lock_and_signal lock;
    bool stop_flag;
    std::queue<std::pair<connect_data *, chan_handle> > connect_queue;
    std::queue<connect_data*> close_connection_queue;
    std::queue<write_data*> write_queue;
    std::queue<read_start_data*> read_start_queue;

public:

    rust_uvtmp_thread() {
	task = rust_scheduler::get_task();
	stop_flag = false;
	loop = uv_loop_new();
	uv_idle_init(loop, &idle);
	idle.data = this;
	uv_idle_start(&idle, idle_cb);
    }

    ~rust_uvtmp_thread() {
	uv_loop_delete(loop);
    }

    void stop() {
	scoped_lock with(lock);
	stop_flag = true;
    }

    connect_data *connect(uint32_t req_id, char *ip, chan_handle chan) {
	scoped_lock with(lock);
        if (req_map.count(req_id)) return NULL;
        connect_data *cd = new connect_data();
        req_map[req_id] = cd;
        cd->req_id = req_id;
        cd->ip_addr = ip;
	connect_queue.push(
            std::pair<connect_data *, chan_handle>(cd, chan));
        return cd;
    }

    void
    close_connection(uint32_t req_id) {
        scoped_lock with(lock);
        connect_data *cd = req_map[req_id];
        close_connection_queue.push(cd);
        req_map.erase(req_id);
    }

    void
    write(uint32_t req_id, uint8_t *buf, size_t len, chan_handle chan) {
        scoped_lock with(lock);
        connect_data *cd = req_map[req_id];
        write_data *wd = new write_data();
        wd->cd = cd;
        wd->buf = new uint8_t[len];
        wd->len = len;
        wd->chan = chan;

        memcpy(wd->buf, buf, len);

        write_queue.push(wd);
    }

    void
    read_start(uint32_t req_id, chan_handle chan) {
        scoped_lock with(lock);
        connect_data *cd = req_map[req_id];
        read_start_data *rd = new read_start_data();
        rd->cd = cd;
        rd->chan = chan;

        read_start_queue.push(rd);
    }

private:

    virtual void
    run() {
	uv_run(loop);
    }

    static void
    idle_cb(uv_idle_t* handle, int status) {
	rust_uvtmp_thread *self = (rust_uvtmp_thread*) handle->data;
	self->on_idle();
    }

    void
    on_idle() {
	scoped_lock with(lock);
	make_new_connections();
	close_connections();
	write_buffers();
	start_reads();
	close_idle_if_stop();
    }

    void
    make_new_connections() {
	assert(lock.lock_held_by_current_thread());
	while (!connect_queue.empty()) {
	    std::pair<connect_data *, chan_handle> pair = connect_queue.front();
	    connect_queue.pop();
            connect_data *cd = pair.first;
	    struct sockaddr_in client_addr = uv_ip4_addr("0.0.0.0", 0);
	    struct sockaddr_in server_addr = uv_ip4_addr(cd->ip_addr, 80);

	    cd->thread = this;
	    cd->chan = pair.second;
	    cd->connect.data = cd;

	    uv_tcp_init(loop, &cd->tcp);
	    uv_tcp_bind(&cd->tcp, client_addr);

	    uv_tcp_connect(&cd->connect, &cd->tcp, server_addr, connect_cb);
	}
    }

    static void
    connect_cb(uv_connect_t *handle, int status) {
	connect_data *cd = (connect_data*)handle->data;
	cd->thread->on_connect(cd);
    }

    void
    on_connect(connect_data *cd) {
	iomsg msg;
	msg.tag = connected_tag;
	msg.val.connected_val = cd;

	send(task, cd->chan, &msg);
    }

    void
    close_connections() {
	assert(lock.lock_held_by_current_thread());
	while (!close_connection_queue.empty()) {
	    connect_data *cd = close_connection_queue.front();
	    close_connection_queue.pop();
	    
	    cd->tcp.data = cd;
	    
	    uv_close((uv_handle_t*)&cd->tcp, tcp_close_cb);
	}
    }

    static void
    tcp_close_cb(uv_handle_t *handle) {
	connect_data *cd = (connect_data*)handle->data;
	delete cd;
    }

    void
    write_buffers() {
	assert(lock.lock_held_by_current_thread());
	while (!write_queue.empty()) {
	    write_data *wd = write_queue.front();
	    write_queue.pop();

	    uv_write_t *write = new uv_write_t();

	    write->data = wd;

	    uv_buf_t buf;
	    buf.base = (char*)wd->buf;
	    buf.len = wd->len;

	    uv_write(write, (uv_stream_t*)&wd->cd->tcp, &buf, 1, write_cb);
	}
    }

    static void
    write_cb(uv_write_t *handle, int status) {
	write_data *wd = (write_data*)handle->data;
	rust_uvtmp_thread *self = wd->cd->thread;
	self->on_write(handle, wd);
    }

    void
    on_write(uv_write_t *handle, write_data *wd) {
	iomsg msg;
	msg.tag = wrote_tag;
	msg.val.wrote_val = wd->cd;

	send(task, wd->chan, &msg);

	delete [] wd->buf;
	delete wd;
	delete handle;
    }

    void
    start_reads() {
	assert (lock.lock_held_by_current_thread());
	while (!read_start_queue.empty()) {
	    read_start_data *rd = read_start_queue.front();
	    read_start_queue.pop();

	    connect_data *cd = rd->cd;
	    cd->tcp.data = rd;

	    uv_read_start((uv_stream_t*)&cd->tcp, alloc_cb, read_cb);
	}
    }

    static uv_buf_t
    alloc_cb(uv_handle_t* handle, size_t size) {
	uv_buf_t buf;
	buf.base = new char[size];
	buf.len = size;
	return buf;
    }

    static void
    read_cb(uv_stream_t *handle, ssize_t nread, uv_buf_t buf) {
	read_start_data *rd = (read_start_data*)handle->data;
	rust_uvtmp_thread *self = rd->cd->thread;
	self->on_read(rd, nread, buf);
    }

    void
    on_read(read_start_data *rd, ssize_t nread, uv_buf_t buf) {
	iomsg msg;
	msg.tag = read_tag;
	msg.val.read_val.cd = rd->cd;
	msg.val.read_val.buf = (uint8_t*)buf.base;
	msg.val.read_val.nread = nread;

	send(task, rd->chan, &msg);
	if (nread == -1) {
	    delete rd;
	}
    }

    void
    close_idle_if_stop() {
	assert(lock.lock_held_by_current_thread());
	if (stop_flag) {
	    uv_close((uv_handle_t*)&idle, NULL);
	}
    }

};

extern "C" rust_uvtmp_thread *
rust_uvtmp_create_thread() {
    rust_uvtmp_thread *thread = new rust_uvtmp_thread();
    return thread;
}

extern "C" void
rust_uvtmp_start_thread(rust_uvtmp_thread *thread) {
    thread->start();    
}

extern "C" void
rust_uvtmp_join_thread(rust_uvtmp_thread *thread) {
    thread->stop();
    thread->join();
}

extern "C" void
rust_uvtmp_delete_thread(rust_uvtmp_thread *thread) {
    delete thread;
}

extern "C" connect_data *
rust_uvtmp_connect(rust_uvtmp_thread *thread, uint32_t req_id, char *ip, chan_handle *chan) {
    return thread->connect(req_id, ip, *chan);
}

extern "C" void
rust_uvtmp_close_connection(rust_uvtmp_thread *thread, uint32_t req_id) {
  thread->close_connection(req_id);
}

extern "C" void
rust_uvtmp_write(rust_uvtmp_thread *thread, uint32_t req_id,
		 uint8_t *buf, size_t len, chan_handle *chan) {
    thread->write(req_id, buf, len, *chan);
}

extern "C" void
rust_uvtmp_read_start(rust_uvtmp_thread *thread, uint32_t req_id,
		      chan_handle *chan) {
    thread->read_start(req_id, *chan);
}

extern "C" void
rust_uvtmp_delete_buf(uint8_t *buf) {
    delete [] buf;
}

extern "C" uint32_t
rust_uvtmp_get_req_id(connect_data *cd) {
    return cd->req_id;
}



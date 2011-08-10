// error-pattern:quux
fn test00_start(ch: chan_t<int>, message: int) {
    send(ch, message);
}

type task_id = int;
type port_id = int;

type chan_t[~T] = {
    task : task_id,
    port : port_id
};

fn send[~T](ch : chan_t<T>, data : -T) { fail; }

fn main() { fail "quux"; }

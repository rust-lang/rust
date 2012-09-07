// error-pattern:quux
fn test00_start(ch: chan_t<int>, message: int) { send(ch, message); }

type task_id = int;
type port_id = int;

enum chan_t<T: Send> = {task: task_id, port: port_id};

fn send<T: Send>(ch: chan_t<T>, data: T) { fail; }

fn main() { fail ~"quux"; }

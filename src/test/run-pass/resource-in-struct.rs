// Ensures that class dtors run if the object is inside an enum
// variant

type closable = @mut bool;

struct close_res {
  i: closable,
 
  drop { *(self.i) = false; }
}

fn close_res(i: closable) -> close_res {
    close_res {
        i: i
    }
}

enum option<T> { none, some(T), }

fn sink(res: option<close_res>) { }

fn main() {
    let c = @mut true;
    sink(none);
    sink(some(close_res(c)));
    assert (!*c);
}

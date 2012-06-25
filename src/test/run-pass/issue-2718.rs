fn sender_terminate<T:send>(p: *packet<T>) {
}

class send_packet<T: send> {
        let mut p: option<*packet<T>>;
        new(p: *packet<T>) { self.p = some(p); }
        drop {
            if self.p != none {
                let mut p = none;
                p <-> self.p;
                sender_terminate(option::unwrap(p))
            }
        }
        fn unwrap() -> *packet<T> {
            let mut p = none;
            p <-> self.p;
            option::unwrap(p)
        }
}

enum state {
        empty,
        full,
        blocked,
        terminated
}

type packet<T: send> = {
        mut state: state,
        mut blocked_task: option<task::task>,
        mut payload: option<T>
};

fn main() {
  let _s: send_packet<int> = send_packet(ptr::addr_of({mut state: empty,
        mut blocked_task: none,
        mut payload: some(42)}));
}

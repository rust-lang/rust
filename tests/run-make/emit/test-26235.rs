// Checks for issue #26235

fn main() {
    use std::thread;

    type Key = u32;
    const NUM_THREADS: usize = 2;

    #[derive(Clone,Copy)]
    struct Stats<S> {
        upsert: S,
        delete: S,
        insert: S,
        update: S
    };

    impl<S> Stats<S> where S: Copy {
        fn dot<B, F, T>(self, s: Stats<T>, f: F) -> Stats<B> where F: Fn(S, T) -> B {
            let Stats { upsert: u1, delete: d1, insert: i1, update: p1 } = self;
            let Stats { upsert: u2, delete: d2, insert: i2, update: p2 } = s;
            Stats { upsert: f(u1, u2), delete: f(d1, d2), insert: f(i1, i2), update: f(p1, p2) }
        }

        fn new(init: S) -> Self {
            Stats { upsert: init, delete: init, insert: init, update: init }
        }
    }

    fn make_threads() -> Vec<thread::JoinHandle<()>> {
        let mut t = Vec::with_capacity(NUM_THREADS);
        for _ in 0..NUM_THREADS {
            t.push(thread::spawn(move || {}));
        }
        t
    }

    let stats = [Stats::new(0); NUM_THREADS];
    make_threads();

    {
        let Stats { ref upsert, ref delete, ref insert, ref update } = stats.iter().fold(
            Stats::new(0), |res, &s| res.dot(s, |x: Key, y: Key| x.wrapping_add(y)));
        println!("upserts: {}, deletes: {}, inserts: {}, updates: {}",
                 upsert, delete, insert, update);
    }
}

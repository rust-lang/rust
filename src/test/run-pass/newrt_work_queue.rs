// xfail-test not a test

pub struct WorkQueue<T> {
    priv queue: ~[T]
}

impl<T> WorkQueue<T> {
    static fn new() -> WorkQueue<T> {
        WorkQueue {
            queue: ~[]
        }
    }

    fn push_back(&mut self, value: T) {
        self.queue.push(value)
    }

    fn pop_back(&mut self) -> Option<T> {
        if !self.queue.is_empty() {
            Some(self.queue.pop())
        } else {
            None
        }
    }

    fn push_front(&mut self, value: T) {
        self.queue.unshift(value)
    }

    fn pop_front(&mut self) -> Option<T> {
        if !self.queue.is_empty() {
            Some(self.queue.shift())
        } else {
            None
        }
    }
}

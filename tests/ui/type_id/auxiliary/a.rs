pub mod a {
    use std::cell::RefCell;

    pub struct Arena<T> {
        chunks: RefCell<ChunkList<T>>,
    }

    struct ChunkList<T> {
        current: Vec<T>,
        rest: Vec<Vec<T>>,
    }

    impl<T> Arena<T> {
        pub fn new() -> Arena<T> {
            Arena {
                chunks: RefCell::new(ChunkList {
                    current: Vec::with_capacity(12),
                    rest: Vec::new(),
                }),
            }
        }
    }
}

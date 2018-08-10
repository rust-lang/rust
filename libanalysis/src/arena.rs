use parking_lot::RwLock;

const CHUNK_LEN: usize = 16;

pub struct Arena<T> {
    chunks: RwLock<Vec<Vec<T>>>,
}

impl<T> Arena<T> {
    pub fn new(&self) -> Arena<T> {
        Arena {
            chunks: RwLock::new(vec![Vec::with_capacity(CHUNK_LEN)]),
        }
    }

    pub fn push(&self, value: T) -> usize {
        let mut guard = self.chunks.write();
        let mut idx = (guard.len() - 1) * CHUNK_LEN;
        let chunk = {
            if guard.last().unwrap().len() == CHUNK_LEN {
                guard.push(Vec::with_capacity(CHUNK_LEN));
            }
            guard.last_mut().unwrap()
        };
        assert!(chunk.len() < chunk.capacity());
        idx += chunk.len();
        chunk.push(value);
        idx
    }

    pub fn get(&self, idx: usize) -> &T {
        let chunk_idx = idx / CHUNK_LEN;
        let chunk_off = idx - chunk_idx * CHUNK_LEN;
        let guard = self.chunks.read();
        let value = &guard[chunk_idx][chunk_off];
        unsafe {
            // We are careful to not move values in chunks,
            // so this hopefully is safe
            ::std::mem::transmute::<&T, &T>(value)
        }
    }
}

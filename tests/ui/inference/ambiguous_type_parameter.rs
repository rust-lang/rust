use std::collections::HashMap;

trait Store<K, V> {
    fn get_raw(&self, key: &K) -> Option<()>;
}

struct InMemoryStore;

impl<K> Store<String, HashMap<K, String>> for InMemoryStore {
    fn get_raw(&self, key: &String) -> Option<()> {
        None
    }
}

fn main() {
    InMemoryStore.get_raw(&String::default()); //~ ERROR type annotations needed
}

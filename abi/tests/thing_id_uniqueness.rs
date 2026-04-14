#[cfg(test)]
mod tests {
    use abi::ThingId;
    use std::collections::HashSet;

    #[test]
    fn test_thing_id_uniqueness() {
        let mut ids = HashSet::new();
        let count = 10_000;

        for _ in 0..count {
            let id = ThingId::new_debug_nonce();
            if ids.contains(&id) {
                panic!("Collision detected! ID {:?} already exists.", id);
            }
            ids.insert(id);
        }

        assert_eq!(ids.len(), count);
    }
}

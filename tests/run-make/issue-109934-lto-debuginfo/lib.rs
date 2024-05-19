extern crate alloc;

#[cfg(test)]
mod tests {
    #[test]
    fn something_alloc() {
        assert_eq!(Vec::<u32>::new(), Vec::<u32>::new());
    }
}

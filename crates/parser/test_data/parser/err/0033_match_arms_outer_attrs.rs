fn foo() {
    match () {
        _ => (),
        _ => (),
        #[cfg(test)]
    }
}

fn find_errors(mut self) {
    let errors: Vec = vec![
        #[debug_format = "A({})"]
        struct A {}
    ];
}

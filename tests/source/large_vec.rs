// See #1470.

impl Environment {
    pub fn new_root() -> Rc<RefCell<Environment>> {
        let mut env = Environment::new();
        let builtin_functions = &[("println",
                                   Function::NativeVoid(CallSign {
                                                            num_params: 0,
                                                            variadic: true,
                                                            param_types: vec![],
                                                        },
                                                        native_println)),
                                  ("run_http_server",
                                   Function::NativeVoid(CallSign {
                                                            num_params: 1,
                                                            variadic: false,
                                                            param_types:
                                                                vec![Some(ConstraintType::Function)],
                                                        },
                                                        native_run_http_server)),
                                  ("len",
                                   Function::NativeReturning(CallSign {
                                                                 num_params: 1,
                                                                 variadic: false,
                                                                 param_types: vec![None],
                                                             },
                                                             native_len))];
    }
}

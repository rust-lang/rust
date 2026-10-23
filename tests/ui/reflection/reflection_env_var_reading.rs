//@ rustc-env: to_read=hello
//@ rustc-env: hello=there
//@ run-pass
#![feature(type_info)]

fn main() {
    const {
        let var_name_to_read = core::mem::type_info::const_env::var_os("to_read").unwrap();
        let var_name_to_read = match core::str::from_utf8(var_name_to_read) {
            Ok(v) => v,
            Err(_) => panic!("to_read is not utf8 encoded")
        };

        let value = core::mem::type_info::const_env::var_os(&var_name_to_read).unwrap();
        assert!(value.len() == 5);
    };
}

// run-pass
// Tests that codegen_path checks whether a
// pattern-bound var is an upvar (when codegenning
// the for-each body)


fn foo(src: usize) {

    match Some(src) {
      Some(src_id) => {
        for _i in 0_usize..10_usize {
            let yyy = src_id;
            assert_eq!(yyy, 0_usize);
        }
      }
      _ => { }
    }
}

pub fn main() { foo(0_usize); }

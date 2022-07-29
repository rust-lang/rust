#![allow(unused_assignments, unused_variables, dead_code)]

fn main() {
    // Initialize test constants in a way that cannot be determined at compile time, to ensure
    // rustc and LLVM cannot optimize out statements (or coverage counters) downstream from
    // dependent conditions.
    let is_true = std::env::args().len() == 1;

    let mut countdown = 0;
    if is_true {
        countdown = 10;
    }

    mod in_mod {
        const IN_MOD_CONST: u32 = 1000;
    }

    fn in_func(a: u32) {
        let b = 1;
        let c = a + b;
        println!("c = {}", c)
    }

    struct InStruct {
        in_struct_field: u32,
    }

    const IN_CONST: u32 = 1234;

    trait InTrait {
        fn trait_func(&mut self, incr: u32);

        fn default_trait_func(&mut self) {
            in_func(IN_CONST);
            self.trait_func(IN_CONST);
        }
    }

    impl InTrait for InStruct {
        fn trait_func(&mut self, incr: u32) {
            self.in_struct_field += incr;
            in_func(self.in_struct_field);
        }
    }

    type InType = String;

    if is_true {
        in_func(countdown);
    }

    let mut val = InStruct {
        in_struct_field: 101,
    };

    val.default_trait_func();
}

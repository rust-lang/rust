#![allow(unused_assignments, unused_variables)]

fn main() {
    // Initialize test constants in a way that cannot be determined at compile time, to ensure
    // rustc and LLVM cannot optimize out statements (or coverage counters) downstream from
    // dependent conditions.
    let is_true = std::env::args().len() == 1;

    let mut countdown = 0;
    if is_true {
        countdown = 10;
    }

    mod inner_mod {
        const INNER_MOD_CONST: u32 = 1000;
    }

    fn inner_function(a: u32) {
        let b = 1;
        let c = a + b;
        println!("c = {}", c)
    }

    struct InnerStruct {
        inner_struct_field: u32,
    }

    const INNER_CONST: u32 = 1234;

    trait InnerTrait {
        fn inner_trait_func(&mut self, incr: u32);

        fn default_trait_func(&mut self) {
            inner_function(INNER_CONST);
            self.inner_trait_func(INNER_CONST);
        }
    }

    impl InnerTrait for InnerStruct {
        fn inner_trait_func(&mut self, incr: u32) {
            self.inner_struct_field += incr;
            inner_function(self.inner_struct_field);
        }
    }

    type InnerType = String;

    if is_true {
        inner_function(countdown);
    }

    let mut val = InnerStruct {
        inner_struct_field: 101,
    };

    val.default_trait_func();
}

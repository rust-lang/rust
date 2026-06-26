//@ normalize-stderr: "DefId\(.+?\)" -> "DefId(..)"
#![feature(rustc_attrs)]

fn bar() {
    fn foo() {
        #[rustc_dump_def_parents]
        fn baz() {
            //~^ ERROR: rustc_dump_def_parents: DefId
            || {
                qux::<
                    {
                        //~^ ERROR: rustc_dump_def_parents: DefId
                        fn inhibits_dump() {
                            qux::<
                                {
                                    //~^ ERROR: rustc_dump_def_parents: DefId
                                    "hi";
                                    1
                                },
                            >();
                        }

                        qux::<{ 1 + 1 }>();
                        //~^ ERROR: rustc_dump_def_parents: DefId
                        1
                    },
                >();
            };
        }
    }
}

const fn qux<const N: usize>() {}

fn main() {}

mod argparse {
    extern mod std;

    use either::{Either, Left, Right};

    struct Flag {
        name: &str,
        desc: &str,
        max_count: uint,
        mut value: uint
    }

    fn flag(name: &r/str, desc: &r/str) -> Flag/&r {
        Flag { name: name, desc: desc, max_count: 1, value: 0 }
    }

    impl Flag {
        fn set_desc(self, s: &str) -> Flag {
            Flag { //~ ERROR cannot infer an appropriate lifetime
                name: self.name,
                desc: s,
                max_count: self.max_count,
                value: self.value
            }
        }
    }
}

fn main () {
    let f : argparse::Flag = argparse::flag(~"flag", ~"My flag");
    let updated_flag = f.set_desc(~"My new flag");
    assert updated_flag.desc == "My new flag";
}

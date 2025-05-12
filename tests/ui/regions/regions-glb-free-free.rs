mod argparse {
    pub struct Flag<'a> {
        name: &'a str,
        pub desc: &'a str,
        max_count: usize,
        value: usize
    }

    pub fn flag<'r>(name: &'r str, desc: &'r str) -> Flag<'r> {
        Flag { name: name, desc: desc, max_count: 1, value: 0 }
    }

    impl<'a> Flag<'a> {
        pub fn set_desc(self, s: &str) -> Flag<'a> {
            Flag { //~ ERROR explicit lifetime required in the type of `s` [E0621]
                name: self.name,
                desc: s,
                max_count: self.max_count,
                value: self.value
            }
        }
    }
}

fn main () {
    let f : argparse::Flag = argparse::flag("flag", "My flag");
    let updated_flag = f.set_desc("My new flag");
    assert_eq!(updated_flag.desc, "My new flag");
}

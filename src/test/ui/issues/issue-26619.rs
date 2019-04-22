#![feature(slice_patterns)]

pub struct History<'a> { pub _s: &'a str }

impl<'a> History<'a> {
    pub fn get_page(&self) {
        for s in vec!["1|2".to_string()].into_iter().filter_map(|ref line| self.make_entry(line)) {
            //~^ ERROR cannot return value referencing function parameter
            println!("{:?}", s);
        }
    }

    fn make_entry(&self, s: &'a String) -> Option<&str> {
        let parts: Vec<_> = s.split('|').collect();
        println!("{:?} -> {:?}", s, parts);

        if let [commit, ..] = &parts[..] { Some(commit) } else { None }
    }
}

fn main() {
    let h = History{ _s: "" };
    h.get_page();
}

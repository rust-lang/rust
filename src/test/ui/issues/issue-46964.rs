// build-pass (FIXME(62277): could be check-pass?)
mod my_mod {
    #[derive(Clone, Copy, Eq, PartialEq, PartialOrd, Ord, Hash)]
    pub struct Name<'a> {
        source: &'a str,
    }

    pub const JSON: Name = Name { source: "JSON" };
}

pub fn crash() -> bool {
  match (my_mod::JSON, None) {
    (_, Some(my_mod::JSON)) => true,
    (my_mod::JSON, None) => true,
    _ => false,
  }
}

fn main() {}

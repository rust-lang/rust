trait Speak : Sized {
    fn say(&self, s:&str) -> String;
    fn hi(&self) -> String { hello(self) }
}

fn hello<S:Speak>(s:&S) -> String{
    s.say("hello")
}

impl Speak for isize {
    fn say(&self, s:&str) -> String {
        format!("{}: {}", s, *self)
    }
}

impl<T: Speak> Speak for Option<T> {
    fn say(&self, s:&str) -> String {
        match *self {
            None => format!("{} - none", s),
            Some(ref x) => { format!("something!{}", x.say(s)) }
        }
    }
}


pub fn main() {
    assert_eq!(3.hi(), "hello: 3".to_string());
    assert_eq!(Some(Some(3)).hi(),
               "something!something!hello: 3".to_string());
    assert_eq!(None::<isize>.hi(), "hello - none".to_string());

    assert_eq!(Some(None::<isize>).hi(), "something!hello - none".to_string());
    assert_eq!(Some(3).hi(), "something!hello: 3".to_string());
}

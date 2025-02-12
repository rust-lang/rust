#![allow(clippy::single_char_pattern)]

struct Issue12068;

impl AsRef<str> for Issue12068 {
    fn as_ref(&self) -> &str {
        ""
    }
}

#[allow(clippy::to_string_trait_impl)]
impl ToString for Issue12068 {
    fn to_string(&self) -> String {
        String::new()
    }
}

fn main() {
    let _ = "a".to_string().split('a').next().unwrap();
    //~^ unnecessary_to_owned

    let _ = "a".to_string().split("a").next().unwrap();
    //~^ unnecessary_to_owned

    let _ = "a".to_owned().split('a').next().unwrap();
    //~^ unnecessary_to_owned

    let _ = "a".to_owned().split("a").next().unwrap();
    //~^ unnecessary_to_owned

    let _ = Issue12068.to_string().split('a').next().unwrap();
    //~^ unnecessary_to_owned

    let _ = [1].to_vec().split(|x| *x == 2).next().unwrap();
    //~^ unnecessary_to_owned

    let _ = [1].to_vec().split(|x| *x == 2).next().unwrap();
    //~^ unnecessary_to_owned

    let _ = [1].to_owned().split(|x| *x == 2).next().unwrap();
    //~^ unnecessary_to_owned

    let _ = [1].to_owned().split(|x| *x == 2).next().unwrap();
    //~^ unnecessary_to_owned
}

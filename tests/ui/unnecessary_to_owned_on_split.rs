#[allow(clippy::single_char_pattern)]

fn main() {
    let _ = "a".to_string().split('a').next().unwrap();
    //~^ ERROR: unnecessary use of `to_string`
    let _ = "a".to_string().split("a").next().unwrap();
    //~^ ERROR: unnecessary use of `to_string`
    let _ = "a".to_owned().split('a').next().unwrap();
    //~^ ERROR: unnecessary use of `to_owned`
    let _ = "a".to_owned().split("a").next().unwrap();
    //~^ ERROR: unnecessary use of `to_owned`

    let _ = [1].to_vec().split(|x| *x == 2).next().unwrap();
    //~^ ERROR: unnecessary use of `to_vec`
    let _ = [1].to_vec().split(|x| *x == 2).next().unwrap();
    //~^ ERROR: unnecessary use of `to_vec`
    let _ = [1].to_owned().split(|x| *x == 2).next().unwrap();
    //~^ ERROR: unnecessary use of `to_owned`
    let _ = [1].to_owned().split(|x| *x == 2).next().unwrap();
    //~^ ERROR: unnecessary use of `to_owned`
}

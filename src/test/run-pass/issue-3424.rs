// rustc --test ignores2.rs && ./ignores2

pub struct Path;

type rsrc_loader = Box<FnMut(&Path) -> Result<String, String>>;

fn tester()
{
    let mut loader: rsrc_loader = Box::new(move |_path| {
        Ok("more blah".to_string())
    });

    let path = Path;
    assert!(loader(&path).is_ok());
}

pub fn main() {}

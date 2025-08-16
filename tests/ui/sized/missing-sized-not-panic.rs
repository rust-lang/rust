// ICE-141806
trait Trait<T> {}
fn func(x: *const dyn Trait<()>)
where
    Missing: Sized, //~ ERROR E0412
{
    let _x: *const dyn Trait<u8> = x as _;
}

fn main() {}

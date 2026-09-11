// rustfmt-brace_style: AlwaysNextLine
// rustfmt-empty_item_single_line: false

fn function()
{
}

struct Struct {}

enum Enum {}

trait Trait
{
}

impl<T> Trait for T
{
}

trait Trait2<T>
where
    T: Copy + Display + Write + Read + FromStr,
{
}

trait Trait3<T>
where
    T: Something
        + SomethingElse
        + Sync
        + Send
        + Display
        + Debug
        + Copy
        + Hash
        + Debug
        + Display
        + Write
        + Read,
{
}

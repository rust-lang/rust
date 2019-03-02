trait Howness {}
impl Howness for () {
    fn how_are_you(&self -> Empty {
    //~^ ERROR expected one of `)` or `,`, found `->`
    //~| ERROR method `how_are_you` is not a member of trait `Howness`
    //~| ERROR cannot find type `Empty` in this scope
        Empty
        //~^ ERROR cannot find value `Empty` in this scope
    }
}
//~^ ERROR expected one of `async`, `const`, `crate`, `default`, `existential`, `extern`, `fn`,

fn main() {}

// ignore-tidy-linelength

// Check that creating/matching on an enum variant through an alias with
// the wrong braced/unit form is caught as an error.

enum Enum { Braced {}, Unit, Tuple() }
type Alias = Enum;

fn main() {
    Alias::Braced;
    //~^ ERROR expected unit struct/variant or constant, found struct variant `<Alias>::Braced` [E0533]
    let Alias::Braced = panic!();
    //~^ ERROR expected unit struct/variant or constant, found struct variant `<Alias>::Braced` [E0533]
    let Alias::Braced(..) = panic!();
    //~^ ERROR expected tuple struct/variant, found struct variant `<Alias>::Braced` [E0164]

    Alias::Unit();
    //~^ ERROR expected function, found enum variant `<Alias>::Unit`
    let Alias::Unit() = panic!();
    //~^ ERROR expected tuple struct/variant, found unit variant `<Alias>::Unit` [E0164]
}

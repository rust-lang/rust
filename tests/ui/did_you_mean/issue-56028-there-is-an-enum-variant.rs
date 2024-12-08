enum PutDown { Set }
enum AffixHeart { Set }
enum CauseToBe { Set }
enum Determine { Set }
enum TableDishesAction { Set }
enum Solidify { Set }
enum UnorderedCollection { Set }

fn setup() -> Set { Set }
//~^ ERROR cannot find type `Set` in this scope
//~| ERROR cannot find value `Set` in this scope

fn main() {
    setup();
}

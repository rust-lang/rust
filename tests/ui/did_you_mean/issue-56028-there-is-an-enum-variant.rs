enum PutDown { Set }
enum AffixHeart { Set }
enum CauseToBe { Set }
enum Determine { Set }
enum TableDishesAction { Set }
enum Solidify { Set }
enum UnorderedCollection { Set }

fn setup() -> Set { Set }
//~^ ERROR cannot find type `Set`
//~| ERROR cannot find value `Set`

fn main() {
    setup();
}

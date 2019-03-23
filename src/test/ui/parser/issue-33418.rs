// run-rustfix

trait Tr: !SuperA {}
//~^ ERROR negative trait bounds are not supported
trait Tr2: SuperA + !SuperB {}
//~^ ERROR negative trait bounds are not supported
trait Tr3: !SuperA + SuperB {}
//~^ ERROR negative trait bounds are not supported
trait Tr4: !SuperA + SuperB
    + !SuperC + SuperD {}
//~^ ERROR negative trait bounds are not supported
trait Tr5: !SuperA
    + !SuperB {}
//~^ ERROR negative trait bounds are not supported

trait SuperA {}
trait SuperB {}
trait SuperC {}
trait SuperD {}

fn main() {}

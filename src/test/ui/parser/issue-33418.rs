// run-rustfix

trait Tr: !SuperA {} //~ ERROR negative trait bounds are not supported
trait Tr2: SuperA + !SuperB {} //~ ERROR negative trait bounds are not supported
trait Tr3: !SuperA + SuperB {} //~ ERROR negative trait bounds are not supported
trait Tr4: !SuperA + SuperB //~ ERROR negative trait bounds are not supported
    + !SuperC + SuperD {}
trait Tr5: !SuperA //~ ERROR negative trait bounds are not supported
    + !SuperB {}

trait SuperA {}
trait SuperB {}
trait SuperC {}
trait SuperD {}

fn main() {}

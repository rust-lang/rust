trait Tr: !SuperA {}
//~^ ERROR negative bounds are not supported
trait Tr2: SuperA + !SuperB {}
//~^ ERROR negative bounds are not supported
trait Tr3: !SuperA + SuperB {}
//~^ ERROR negative bounds are not supported
trait Tr4: !SuperA + SuperB
//~^ ERROR negative bounds are not supported
+ !SuperC + SuperD {}
//~^ ERROR negative bounds are not supported
trait Tr5: !SuperA
//~^ ERROR negative bounds are not supported
+ !SuperB {}
//~^ ERROR negative bounds are not supported

trait SuperA {}
trait SuperB {}
trait SuperC {}
trait SuperD {}

fn main() {}

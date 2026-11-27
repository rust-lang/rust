// Ensure that we detect and reject implicit forward references to generic parameters in type
// parameter defaults arising from type-relative associated item paths.

// Below, `Self::Type` resolves to `<Self as Def0<T>>::Type` but `T` is a forward reference.
// issue: <https://github.com/rust-lang/rust/issues/165989>
trait Def0<T = Self::Type> { type Type; }
//~^ ERROR type parameter default references generic parameter `T` before it is declared
fn use0(_: impl Def0) {}

// Below, `T::Type` resolves to `<T as Trait<U>>::Type` but `U` is a forward reference.
struct Def1<T: Trait<U>, U = T::Type>(T, U);
//~^ ERROR type parameter default references generic parameter `U` before it is declared
struct Use1(Def1<()>);

// Below, `T::Type` resolves to `<T as Trait<[V; N]>>::Type` but `V` & `N` are forward references.
struct Def2<T: Trait<[V; N]>, U = T::Type, V = (), const N: usize = 0>(T, U, V);
//~^ ERROR type parameter default references generic parameter `V` before it is declared
//~| ERROR type parameter default references generic parameter `N` before it is declared
struct Use2(Def2<()>);

trait Trait<T> { type Type; }

fn main() {}

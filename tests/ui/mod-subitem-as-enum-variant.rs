mod Mod {
    pub struct FakeVariant<T>(pub T);
}

fn main() {
    Mod::FakeVariant::<i32>(0);
    Mod::<i32>::FakeVariant(0);
    //~^ ERROR type arguments are not allowed on module `Mod` [E0109]
}

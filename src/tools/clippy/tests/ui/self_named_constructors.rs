#![warn(clippy::self_named_constructors)]

struct ShouldSpawn;
struct ShouldNotSpawn;

impl ShouldSpawn {
    pub fn should_spawn() -> ShouldSpawn {
        //~^ self_named_constructors

        ShouldSpawn
    }

    fn should_not_spawn() -> ShouldNotSpawn {
        ShouldNotSpawn
    }
}

impl ShouldNotSpawn {
    pub fn new() -> ShouldNotSpawn {
        ShouldNotSpawn
    }
}

struct ShouldNotSpawnWithTrait;

trait ShouldNotSpawnTrait {
    type Item;
}

impl ShouldNotSpawnTrait for ShouldNotSpawnWithTrait {
    type Item = Self;
}

impl ShouldNotSpawnWithTrait {
    pub fn should_not_spawn_with_trait() -> impl ShouldNotSpawnTrait<Item = Self> {
        ShouldNotSpawnWithTrait
    }
}

// Same trait name and same type name should not spawn the lint
#[derive(Default)]
pub struct Default;

trait TraitSameTypeName {
    fn should_not_spawn() -> Self;
}
impl TraitSameTypeName for ShouldNotSpawn {
    fn should_not_spawn() -> Self {
        ShouldNotSpawn
    }
}

struct SelfMethodShouldNotSpawn;

impl SelfMethodShouldNotSpawn {
    fn self_method_should_not_spawn(self) -> Self {
        SelfMethodShouldNotSpawn
    }
}

fn main() {}

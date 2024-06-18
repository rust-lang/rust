trait ChooseMe {
    type Type;
}

trait PickMe {
    type Type;
}

trait HaveItAll {
    type See: ChooseMe + PickMe;
}

struct Env<T: HaveItAll>(T::See::Type);
//~^ ERROR ambiguous associated type `Type` in bounds of `<T as HaveItAll>::See`

fn main() {}

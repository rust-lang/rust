trait Super {
    type Type;
}
trait Sub1: Super {
    auto impl Super;
}

trait Sub2: Super {
    auto impl Super {}
}

trait Sub3: Super {
    auto impl Super {
        type Type = u8;
    }
}

impl Sub1 for () {
    auto impl Super {
        type Type = u8;
    }
}

impl Sub2 for () {
    auto impl Super {
        type Type = u8;
    }
}

impl Sub3 for () {
    auto impl Super;
}

impl Sub3 for (u8,) {
    auto impl Super {
        type Type = u8;
    }
}

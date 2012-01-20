enum maybe_pointy {
    no_pointy,
    yes_pointy(@pointy),
}

type pointy = {
    mutable x : maybe_pointy
};

fn main() {
    let m = @{ mutable x : no_pointy };
    m.x = yes_pointy(m);
}

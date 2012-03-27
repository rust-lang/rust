enum maybe_pointy {
    no_pointy,
    yes_pointy(@pointy),
}

type pointy = {
    mut x : maybe_pointy
};

fn main() {
    let m = @{ mut x : no_pointy };
    m.x = yes_pointy(m);
}

trait Wedding<'t>: 't { }

struct Prince<'kiss, 'SnowWhite> {
    child: Box<dyn Wedding<'kiss> + 'SnowWhite>, //~ ERROR E0478
}

fn main() {
}

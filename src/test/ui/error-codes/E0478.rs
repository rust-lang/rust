trait Wedding<'t>: 't { }

struct Prince<'kiss, 'SnowWhite> {
    child: Box<Wedding<'kiss> + 'SnowWhite>, //~ ERROR E0478
}

fn main() {
}

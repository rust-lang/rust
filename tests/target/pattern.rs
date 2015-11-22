fn main() {
    let z = match x {
        "pat1" => 1,
        (ref x, ref mut y /* comment */) => 2,
    };

    if let <T as Trait>::CONST = ident {
        do_smth();
    }

    let Some(ref xyz /* comment! */) = opt;

    if let None = opt2 {
        panic!("oh noes");
    }
}

impl<'a, 'b> ResolveGeneratedContentFragmentMutator<'a, 'b> {
    fn mutate_fragment(&mut self, fragment: &mut Fragment) {
        match **info {
            GeneratedContentInfo::ContentItem(ContentItem::Counter(ref counter_name,
                                                                   counter_style)) => {}
        }
    }
}

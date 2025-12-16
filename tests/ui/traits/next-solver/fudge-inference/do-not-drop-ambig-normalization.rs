//@ compile-flags: -Znext-solver
//@ check-pass

// A regression test for https://github.com/rust-lang/trait-system-refactor-initiative/issues/259.
// We previously normalized inside of a call to `fn fudge_inference_if_ok`. If that normalization
// ended up ambiguous, we'd drop the normalization goal and return an unconstrained infer var.
//
// This meant that even though `DB::MetadataLookup` could be normalized after equating the
// receiver with the self type, at this point the normalization goal was no longer around.

trait BindCollector<DB: Backend> {
    fn push_bound_value(self, metadata_lookup: &DB::MetadataLookup);
}

trait Backend {
    type BindCollector;
    type MetadataLookup;
}

fn foo<DB: Backend<BindCollector: BindCollector<DB>>>(
    collector: DB::BindCollector,
    metadata_lookup: &&DB::MetadataLookup,
) {
    collector.push_bound_value(metadata_lookup)
}

fn main() {}

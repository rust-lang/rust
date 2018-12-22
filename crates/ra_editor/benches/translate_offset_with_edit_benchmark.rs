use criterion::{criterion_group, criterion_main};
use criterion::Criterion;
use criterion::Fun;
use ra_text_edit::AtomTextEdit;
use ra_text_edit::test_utils::{arb_edits_custom, arb_offset};
use ra_editor::line_index_utils;
use ra_editor::LineIndex;
use ra_syntax::TextUnit;
use proptest::test_runner;
use proptest::string::string_regex;
use proptest::strategy::{Strategy, ValueTree};
use rand_xorshift::XorShiftRng;
use rand::SeedableRng;
use lazy_static::lazy_static;

#[derive(Debug)]
struct Data {
    text: String,
    line_index: LineIndex,
    edits: Vec<AtomTextEdit>,
    offset: TextUnit,
}

fn setup_data() -> Data {
    let mut runner = test_runner::TestRunner::default();
    {
        struct TestRng {
            rng: XorShiftRng,
        }
        // HACK to be able to manually seed the TestRunner
        let rng: &mut TestRng = unsafe { std::mem::transmute(runner.rng()) };
        rng.rng = XorShiftRng::seed_from_u64(0);
    }

    let text = {
        let arb = string_regex("([a-zA-Z_0-9]{10,50}.{1,5}\n){100,500}").unwrap();
        let tree = arb.new_tree(&mut runner).unwrap();
        tree.current()
    };

    let edits = {
        let arb = arb_edits_custom(&text, 99, 100);
        let tree = arb.new_tree(&mut runner).unwrap();
        tree.current()
    };

    let offset = {
        let arb = arb_offset(&text);
        let tree = arb.new_tree(&mut runner).unwrap();
        tree.current()
    };

    let line_index = LineIndex::new(&text);

    Data {
        text,
        line_index,
        edits,
        offset,
    }
}

lazy_static! {
    static ref DATA: Data = setup_data();
}

fn compare_translates(c: &mut Criterion) {
    let functions = vec![
        Fun::new("translate_after_edit", |b, _| {
            b.iter(|| {
                let d = &*DATA;
                line_index_utils::translate_after_edit(&d.text, d.offset, d.edits.clone());
            })
        }),
        Fun::new("translate_offset_with_edit", |b, _| {
            b.iter(|| {
                let d = &*DATA;
                line_index_utils::translate_offset_with_edit(&d.line_index, d.offset, &d.edits);
            })
        }),
    ];

    c.bench_functions("translate", functions, ());
}

criterion_group!(benches, compare_translates);
criterion_main!(benches);

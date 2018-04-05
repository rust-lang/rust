// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::*;
use itertools::Itertools;
use std::collections::HashMap;

use test::{CombineTest, TestSettings};
use testing::{capture_output, ShouldPanic};
use testing::{TestDesc, TestDescAndFn, TestEvent, TestFn, TestName, TestResult};

use rustc_driver;
use std::fs::File;
use std::io::Seek;
use std::io::SeekFrom;
use std::panic;
use std::sync::Arc;
use std::sync::Mutex;
use std::thread;
use std::thread::JoinHandle;
use std::time::Instant;
use syntax::with_globals;
use tempdir::TempDir;

use syntax_pos::FileName;

#[derive(Eq, PartialEq, Hash, Clone, Debug)]
struct TestGroup {
    features: Vec<String>,
    as_test_harness: bool,
    no_run: bool,
}

pub fn test_combined(
    all_tests: Vec<TestDescAndFn>,
    _threads: usize,
) -> Option<JoinHandle<Vec<TestEvent>>> {
    //eprintln!("running rustdoc0 combined - {} tests", all_tests.len());
    Some(thread::spawn(move || {
        //eprintln!("running rustdoc combined - {} tests", all_tests.len());

        if all_tests.is_empty() {
            return Vec::new();
        }

        let mut feature_groups: HashMap<TestGroup, Vec<(TestDesc, CombineTest)>> = HashMap::new();

        let mut settings = None;
        let mut events = Vec::new();
        let mut test_count = 0;

        for mut test in all_tests {
            let data = match test.testfn {
                TestFn::CombineTest(ref mut t) => {
                    if test.desc.ignore {
                        None
                    } else {
                        t.downcast_mut::<Option<CombineTest>>().unwrap().take()
                    }
                }
                _ => panic!(),
            };
            match data {
                Some(data) => {
                    settings = Some(data.settings.clone());
                    let group = TestGroup {
                        features: data.features.clone(),
                        as_test_harness: data.as_test_harness,
                        no_run: data.no_run,
                    };
                    test_count += 1;
                    feature_groups
                        .entry(group)
                        .or_insert(Vec::new())
                        .push((test.desc, data));
                }
                None => {
                    events.push(TestEvent::TeResult(
                        test.desc,
                        TestResult::TrIgnored,
                        Vec::new(),
                    ));
                }
            }
        }

        let settings = settings.unwrap();

        let feature_groups: Vec<(TestGroup, Vec<(TestDesc, CombineTest)>)> =
            feature_groups.into_iter().collect();

        let threads = 16;
        let largest_group = (test_count / threads) + threads - 1;

        // Split large groups into smaller ones
        let mut feature_groups: Vec<(TestGroup, Vec<(TestDesc, CombineTest)>)> = feature_groups
            .into_iter()
            .flat_map(|(f, group)| {
                let chunks = group.into_iter().chunks(largest_group);
                let groups: Vec<_> = chunks
                    .into_iter()
                    .map(|chunk| {
                        let group: Vec<_> = chunk.into_iter().collect();
                        (f.clone(), group)
                    })
                    .collect();
                groups
            })
            .collect();

        // Run largest groups first
        feature_groups.sort_by_key(|a: &(TestGroup, Vec<(TestDesc, CombineTest)>)| a.1.len());
        /*
        for &(ref f, ref group) in &feature_groups {
            eprintln!("group [{:?}] has {} tests", f, group.len());
        }
*/
        let groups = Arc::new(Mutex::new(feature_groups));

        let threads: Vec<_> = (0..threads)
            .map(|i| {
                let groups = groups.clone();
                let settings = settings.clone();
                thread::spawn(move || {
                    let mut events = Vec::new();
                    while let Some((features, group)) = {
                        let mut lock = groups.lock().unwrap();
                        lock.pop()
                    } {
                        let results = run_combined_instance(&settings, i, features, group);
                        events.extend(results);
                    }
                    events
                })
            })
            .collect();
        events.extend(
            threads
                .into_iter()
                .flat_map(|thread| thread.join().unwrap()),
        );
        events
    }))
}

fn run_combined_instance(
    settings: &Arc<TestSettings>,
    instance: usize,
    group: TestGroup,
    tests: Vec<(TestDesc, CombineTest)>,
) -> Vec<TestEvent> {
    let mut events = Vec::new();
    let outdir = TempDir::new("rustdoctest")
        .ok()
        .expect("rustdoc needs a tempdir");
    let progress_file = outdir.as_ref().join("progress");

    let mut out = String::new();
    for feature in &group.features {
        out.push_str(&format!("#![feature({})]\n", feature));
    }

    out.push_str(
        "#![allow(warnings)]
",
    );

    for (i, test) in tests.iter().enumerate() {
        let (test, _) = test::make_test(
            &test.1.test,
            Some(&settings.cratename),
            test.1.as_test_harness,
            &settings.opts,
        );
        out.push_str(&format!(
            "pub mod _combined_test_{} {{ {} }}\n",
            i, test
        ));
    }

   if !group.as_test_harness {
    out.push_str(
        "fn main() {
        use std::fs::File;
        use std::io::Read;
    ",
    );
    out.push_str(&format!(
        "\
         let mut file = File::open({:?}).unwrap();",
        progress_file
    ));
    out.push_str(
        "
        let mut c = String::new();
        file.read_to_string(&mut c).unwrap();
        let mut i = c.parse::<usize>().unwrap();
        match i {
    ",
    );

    for i in 0..tests.len() {
        out.push_str(&format!(
            "\
             {} => {{ let _: () = _combined_test_{}::main(); }},\n",
            i, i
        ));
    }

    out.push_str(
        "\
         _ => panic!(\"unknown test\")\n        }\n}\n",
    );
   }
    /*
    props.compile_flags.push("-C".to_string());
    props.compile_flags.push("codegen-units=1".to_string());
    props.compile_flags.push("-A".to_string());
    props.compile_flags.push("warnings".to_string());
    props.compile_flags.push("-Z".to_string());
    props.compile_flags.push("combine-tests".to_string());

*/
    let start = Instant::now();

    let compile_task = TestDesc {
        name: TestName::StaticTestName("combined compilation of tests"),
        ignore: false,
        should_panic: ShouldPanic::No,
        allow_fail: false,
    };

    let mut compiled = None;
    let (result, output) = capture_output(&compile_task, false /* FIXME */, || {
        let panic = io::set_panic(None);
        let print = io::set_print(None);
        let settings = settings.clone();
        let group = group.clone();
        let outdir = outdir.as_ref().to_path_buf();
        match {
            rustc_driver::in_rustc_thread(move || {
                with_globals(move || {
                    io::set_panic(panic);
                    io::set_print(print);
                    test::compile_test(
                        out,
                        &FileName::Anon,
                        0,
                        &settings.cfgs,
                        settings.libs.clone(),
                        settings.externs.clone(),
                        group.as_test_harness,
                        false,
                        group.no_run,
                        Vec::new(), // FIXME: error_codes,
                        settings.maybe_sysroot.clone(),
                        settings.linker.clone(),
                        &outdir,
                        0,
                        true,
                    )
                })
            })
        } {
            | Ok(data) => compiled = Some(data),
            Err(err) => panic::resume_unwind(err),
        }
    });

    if result != TestResult::TrOk {
        events.push(TestEvent::TeCombinedFail(
            compile_task,
            result,
            output,
            tests.len(),
        ));
        return events;
    }

    let libdir = compiled.unwrap();

    println!("output {:?}", outdir);

    let time = Instant::now().duration_since(start);
    /*println!(
        "rustdoc combined test {} compiled in {} seconds",
        instance,
        time.as_secs()
    );*/

    if group.no_run {
        for test in tests {
            events.push(TestEvent::TeResult(test.0, TestResult::TrOk, Vec::new()));
        }
        return events;
    }

    let start = Instant::now();

    let mut progress = File::create(&progress_file).unwrap();

    // FIXME: Setup exec-env

    for (i, test) in tests.into_iter().enumerate() {
        progress.seek(SeekFrom::Start(0)).unwrap();
        progress.set_len(0).unwrap();
        progress.write_all(format!("{}", i).as_bytes()).unwrap();
        progress.flush().unwrap();

        let (result, output) = capture_output(&test.0, false, || {
            test::run_built_test(outdir.as_ref(), &libdir, test.1.should_panic)
        });

        events.push(TestEvent::TeResult(test.0, result, output));
    }

    let time = Instant::now().duration_since(start);
    /*println!(
        "rustdoc combined test {} ran in {} seconds",
        instance,
        time.as_secs()
    );*/

    //::std::mem::drop(outdir.into_path());

    events
}

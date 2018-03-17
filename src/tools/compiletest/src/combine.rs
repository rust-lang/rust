// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::*;
use common::{CombineTest, Config, TestPaths};
use common::RunPass;
use header::TestProps;
use itertools::Itertools;

use test::{capture_output, ShouldPanic};
use test::{TestDesc, TestName, TestFn, TestDescAndFn, TestResult, TestEvent};

use std::fs::File;
use std::io::SeekFrom;
use std::sync::Arc;
use std::sync::Mutex;
use std::thread;
use std::thread::JoinHandle;
use std::time::SystemTime;
use std::path::{PathBuf};

pub fn run_combined(config: Arc<Config>, all_tests: Vec<TestDescAndFn>,
                    _threads: usize) -> Option<JoinHandle<Vec<TestEvent>>> {
    if config.mode != RunPass {
        assert!(all_tests.is_empty());
        return None;
    }
    Some(thread::spawn(move || {
        let mut feature_groups: HashMap<Vec<String>, Vec<(TestDesc, CombineTest)>> = HashMap::new();

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
                    test_count += 1;
                    feature_groups.entry(data.features.clone())
                                  .or_insert(Vec::new()).push((test.desc, data));
                }
                None => {
                    events.push(TestEvent::TeResult(test.desc, TestResult::TrIgnored, Vec::new()));
                }
            }
        }

        let feature_groups: Vec<(Vec<String>, Vec<(TestDesc, CombineTest)>)> =
            feature_groups.into_iter().collect();

        let threads = 16;
        let largest_group = (test_count / threads) + threads - 1;

        // Split large groups into smaller ones
        let mut feature_groups: Vec<(Vec<String>, Vec<(TestDesc, CombineTest)>)> =
            feature_groups.into_iter().flat_map(|(f, group)| {
                let chunks = group.into_iter().chunks(largest_group);
                let groups: Vec<_> = chunks.into_iter().map(|chunk| {
                    let group: Vec<_> = chunk.into_iter().collect();
                    (f.clone(), group)
                }).collect();
                groups
        }).collect();

        // Run largest groups first
        feature_groups.sort_by_key(|a: &(Vec<String>, Vec<(TestDesc, CombineTest)>)| a.1.len());

        /*for &(ref f, ref group) in &feature_groups {
            eprintln!("features [{:?}] has {} tests", f, group.len());
        }*/

        let groups = Arc::new(Mutex::new(feature_groups));

        let threads: Vec<_> = (0..threads).map(|i| {
            let groups = groups.clone();
            let config = config.clone();
            thread::spawn(move || {
                let mut events = Vec::new();
                while let Some((features, group)) = {
                    let mut lock = groups.lock().unwrap();
                    /*let r = */lock.pop()
                    //drop(lock);
                    //r
                } {
                    let results = run_combined_instance(&*config, i, features, group);
                    events.extend(results);
                }
                events
            })
        }).collect();
        events.extend(threads.into_iter().flat_map(|thread| {
            thread.join().unwrap()
        }));
        events
    }))
}

pub fn run_combined_instance(config: &Config,
                    instance: usize,
                    features: Vec<String>,
                    tests: Vec<(TestDesc, CombineTest)>) -> Vec<TestEvent> {
    let mut events = Vec::new();

    let file = config.build_base.join(format!("run-pass-{}.rs", instance));
    let progress_file = config.build_base.join(format!("run-pass-progress-{}", instance));

    let mut input = File::create(&file).unwrap();

    let mut out = String::new();
    for feature in &features {
        out.push_str(&format!("#![feature({})]\n", feature));
    }

    out.push_str("#![allow(warnings)]
//extern crate core;
");

    for (i, test) in tests.iter().enumerate() {
        out.push_str(&format!("#[path={:?}]\npub mod _combined_test_{};\n",
                              &test.1.paths.file.to_str().unwrap(), i));
    }

    out.push_str("fn main() {
        use std::fs::File;
        use std::io::Read;
    ");
    out.push_str(&format!("\
        let mut file = File::open({:?}).unwrap();", progress_file));
    out.push_str("
        let mut c = String::new();
        file.read_to_string(&mut c).unwrap();
        let mut i = c.parse::<usize>().unwrap();
        match i {
    ");

    for i in 0..tests.len() {
        out.push_str(&format!("\
            {} => {{ let _: () = _combined_test_{}::main(); }},\n", i, i));
    }

    out.push_str("\
            _ => panic!(\"unknown test\")\n        }\n}\n");

    input.write_all(out.as_bytes()).unwrap();
    input.flush().unwrap();

    let paths = TestPaths {
        file: file,
        base: config.src_base.clone(),
        relative_dir: PathBuf::from("."),
    };

    let mut props = TestProps::new();
    props.compile_flags.push("-C".to_string());
    props.compile_flags.push("codegen-units=1".to_string());
    props.compile_flags.push("-A".to_string());
    props.compile_flags.push("warnings".to_string());
    props.compile_flags.push("-Z".to_string());
    props.compile_flags.push("combine-tests".to_string());

    let base_cx = TestCx {
        config: &config,
        props: &props,
        testpaths: &paths,
        revision: None,
        long_compile: true,
    };

    let start = SystemTime::now();

    let compile_task = TestDesc {
        name: TestName::StaticTestName("combined compilation of tests"),
        ignore: false,
        should_panic: ShouldPanic::No,
        allow_fail: false,
    };

    let (result, output) = capture_output(&compile_task, config.no_capture, || {
        base_cx.compile_rpass_test();
    });

    if result != TestResult::TrOk {
        events.push(TestEvent::TeCombinedFail(compile_task, result, output, tests.len()));
        return events;
    }

    //let time = SystemTime::now().duration_since(start).unwrap();
    //println!("run-pass combined test {} compiled in {} seconds", instance, time.as_secs());

    //let start = SystemTime::now();

    let mut progress = File::create(&progress_file).unwrap();

    // FIXME: Setup exec-env

    for (i, test) in tests.into_iter().enumerate() {
        let base_cx = TestCx {
            config: &config,
            props: &test.1.props,
            testpaths: &paths,
            revision: None,
            long_compile: true,
        };

        progress.seek(SeekFrom::Start(0)).unwrap();
        progress.set_len(0).unwrap();
        progress.write_all(format!("{}", i).as_bytes()).unwrap();
        progress.flush().unwrap();

        let (result, output) = capture_output(&test.0, config.no_capture, || {
            let proc_res = base_cx.exec_compiled_test();
            if !proc_res.status.success() {
                base_cx.fatal_proc_rec("test run failed!", &proc_res);
            }
            File::create(::stamp(&config, &test.1.paths)).unwrap();
        });

        events.push(TestEvent::TeResult(test.0, result, output));
    }

    // delete the executable after running it to save space.
     // it is ok if the deletion failed.
    let _ = fs::remove_file(base_cx.make_exe_name());

    //let time = SystemTime::now().duration_since(start).unwrap();
    //println!("run-pass combined test {} ran in {} seconds", instance, time.as_secs());

    events
}

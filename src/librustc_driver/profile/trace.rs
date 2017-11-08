// Copyright 2012-2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::*;
use syntax_pos::SpanData;
use rustc::ty::maps::QueryMsg;
use std::fs::File;
use std::time::{Duration, Instant};
use std::collections::hash_map::HashMap;
use rustc::dep_graph::{DepNode};

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Query {
    pub span: SpanData,
    pub msg: QueryMsg,
}
pub enum Effect {
    QueryBegin(Query, CacheCase),
    TimeBegin(String),
    TaskBegin(DepNode),
}
pub enum CacheCase {
    Hit, Miss
}
/// Recursive trace structure
pub struct Rec {
    pub effect: Effect,
    pub start: Instant,
    pub dur_self: Duration,
    pub dur_total: Duration,
    pub extent: Box<Vec<Rec>>,
}
pub struct QueryMetric {
    pub count: usize,
    pub dur_self: Duration,
    pub dur_total: Duration,
}

pub fn cons_of_query_msg(q: &trace::Query) -> String {
    let s = format!("{:?}", q.msg);
    let cons: Vec<&str> = s.split(|d| d == '(' || d == '{').collect();
    assert!(cons.len() > 0 && cons[0] != "");
    cons[0].to_string()
}

pub fn cons_of_key(k: &DepNode) -> String {
    let s = format!("{:?}", k);
    let cons: Vec<&str> = s.split(|d| d == '(' || d == '{').collect();
    assert!(cons.len() > 0 && cons[0] != "");
    cons[0].to_string()
}

// First return value is text; second return value is a CSS class
pub fn html_of_effect(eff: &Effect) -> (String, String) {
    match *eff {
        Effect::TimeBegin(ref msg) => {
            (msg.clone(),
             format!("time-begin"))
        },
        Effect::TaskBegin(ref key) => {
            let cons = cons_of_key(key);
            (cons.clone(), format!("{} task-begin", cons))
        },
        Effect::QueryBegin(ref qmsg, ref cc) => {
            let cons = cons_of_query_msg(qmsg);
            (cons.clone(),
             format!("{} {}",
                     cons,
                     match *cc {
                         CacheCase::Hit => "hit",
                         CacheCase::Miss => "miss",
                     }))
        }
    }
}

// First return value is text; second return value is a CSS class
fn html_of_duration(_start: &Instant, dur: &Duration) -> (String, String) {
    use rustc::util::common::duration_to_secs_str;
    (duration_to_secs_str(dur.clone()),
     "".to_string()
    )
}

fn html_of_fraction(frac: f64) -> (String, String) {
    let css = {
        if       frac > 0.50  { format!("frac-50") }
        else if  frac > 0.40  { format!("frac-40") }
        else if  frac > 0.30  { format!("frac-30") }
        else if  frac > 0.20  { format!("frac-20") }
        else if  frac > 0.10  { format!("frac-10") }
        else if  frac > 0.05  { format!("frac-05") }
        else if  frac > 0.02  { format!("frac-02") }
        else if  frac > 0.01  { format!("frac-01") }
        else if  frac > 0.001 { format!("frac-001") }
        else                  { format!("frac-0") }
    };
    let percent = frac * 100.0;
    if percent > 0.1 { (format!("{:.1}%", percent), css) }
    else { (format!("< 0.1%", ), css) }
}

fn total_duration(traces: &Vec<Rec>) -> Duration {
    let mut sum : Duration = Duration::new(0,0);
    for t in traces.iter() {
        sum += t.dur_total;
    }
    return sum
}

fn duration_div(nom: Duration, den: Duration) -> f64 {
    fn to_nanos(d: Duration) -> u64 {
        d.as_secs() * 1_000_000_000 + d.subsec_nanos() as u64
    }

    to_nanos(nom) as f64 / to_nanos(den) as f64
}

fn write_traces_rec(file: &mut File, traces: &Vec<Rec>, total: Duration, depth: usize) {
    for t in traces {
        let (eff_text, eff_css_classes) = html_of_effect(&t.effect);
        let (dur_text, dur_css_classes) = html_of_duration(&t.start, &t.dur_total);
        let fraction = duration_div(t.dur_total, total);
        let percent = fraction * 100.0;
        let (frc_text, frc_css_classes) = html_of_fraction(fraction);
        write!(file, "<div class=\"trace depth-{} extent-{}{} {} {} {}\">\n",
               depth,
               t.extent.len(),
               /* Heuristic for 'important' CSS class: */
               if t.extent.len() > 5 || percent >= 1.0 {
                   " important" }
               else { "" },
               eff_css_classes,
               dur_css_classes,
               frc_css_classes,
        ).unwrap();
        write!(file, "<div class=\"eff\">{}</div>\n", eff_text).unwrap();
        write!(file, "<div class=\"dur\">{}</div>\n", dur_text).unwrap();
        write!(file, "<div class=\"frc\">{}</div>\n", frc_text).unwrap();
        write_traces_rec(file, &t.extent, total, depth + 1);
        write!(file, "</div>\n").unwrap();
    }
}

fn compute_counts_rec(counts: &mut HashMap<String,QueryMetric>, traces: &Vec<Rec>) {
    for t in traces.iter() {
        match t.effect {
            Effect::TimeBegin(ref msg) => {
                let qm = match counts.get(msg) {
                    Some(_qm) => { panic!("TimeBegin with non-unique, repeat message") }
                    None => QueryMetric{
                        count: 1,
                        dur_self: t.dur_self,
                        dur_total: t.dur_total,
                    }};
                counts.insert(msg.clone(), qm);
            },
            Effect::TaskBegin(ref key) => {
                let cons = cons_of_key(key);
                let qm = match counts.get(&cons) {
                    Some(qm) =>
                        QueryMetric{
                            count: qm.count + 1,
                            dur_self: qm.dur_self + t.dur_self,
                            dur_total: qm.dur_total + t.dur_total,
                        },
                    None => QueryMetric{
                        count: 1,
                        dur_self: t.dur_self,
                        dur_total: t.dur_total,
                    }};
                counts.insert(cons, qm);
            },
            Effect::QueryBegin(ref qmsg, ref _cc) => {
                let qcons = cons_of_query_msg(qmsg);
                let qm = match counts.get(&qcons) {
                    Some(qm) =>
                        QueryMetric{
                            count: qm.count + 1,
                            dur_total: qm.dur_total + t.dur_total,
                            dur_self: qm.dur_self + t.dur_self
                        },
                    None => QueryMetric{
                        count: 1,
                        dur_total: t.dur_total,
                        dur_self: t.dur_self,
                    }
                };
                counts.insert(qcons, qm);
            }
        }
        compute_counts_rec(counts, &t.extent)
    }
}

pub fn write_counts(count_file: &mut File, counts: &mut HashMap<String,QueryMetric>) {
    use rustc::util::common::duration_to_secs_str;
    use std::cmp::Ordering;

    let mut data = vec![];
    for (ref cons, ref qm) in counts.iter() {
        data.push((cons.clone(), qm.count.clone(), qm.dur_total.clone(), qm.dur_self.clone()));
    };
    data.sort_by(|&(_,_,_,self1),&(_,_,_,self2)|
                 if self1 > self2 { Ordering::Less } else { Ordering::Greater } );
    for (cons, count, dur_total, dur_self) in data {
        write!(count_file, "{}, {}, {}, {}\n",
               cons, count,
               duration_to_secs_str(dur_total),
               duration_to_secs_str(dur_self)
        ).unwrap();
    }
}

pub fn write_traces(html_file: &mut File, counts_file: &mut File, traces: &Vec<Rec>) {
    let mut counts : HashMap<String,QueryMetric> = HashMap::new();
    compute_counts_rec(&mut counts, traces);
    write_counts(counts_file, &mut counts);

    let total : Duration = total_duration(traces);
    write_traces_rec(html_file, traces, total, 0)
}

pub fn write_style(html_file: &mut File) {
    write!(html_file,"{}", "
body {
    font-family: sans-serif;
    background: black;
}
.trace {
    color: black;
    display: inline-block;
    border-style: solid;
    border-color: red;
    border-width: 1px;
    border-radius: 5px;
    padding: 0px;
    margin: 1px;
    font-size: 0px;
}
.task-begin {
    border-width: 1px;
    color: white;
    border-color: #ff8;
    font-size: 0px;
}
.miss {
    border-color: red;
    border-width: 1px;
}
.extent-0 {
    padding: 2px;
}
.time-begin {
    border-width: 4px;
    font-size: 12px;
    color: white;
    border-color: #afa;
}
.important {
    border-width: 3px;
    font-size: 12px;
    color: white;
    border-color: #f77;
}
.hit {
    padding: 0px;
    border-color: blue;
    border-width: 3px;
}
.eff {
  color: #fff;
  display: inline-block;
}
.frc {
  color: #7f7;
  display: inline-block;
}
.dur {
  display: none
}
.frac-50 {
  padding: 10px;
  border-width: 10px;
  font-size: 32px;
}
.frac-40 {
  padding: 8px;
  border-width: 8px;
  font-size: 24px;
}
.frac-30 {
  padding: 6px;
  border-width: 6px;
  font-size: 18px;
}
.frac-20 {
  padding: 4px;
  border-width: 6px;
  font-size: 16px;
}
.frac-10 {
  padding: 2px;
  border-width: 6px;
  font-size: 14px;
}
").unwrap();
}

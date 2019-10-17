//! Module containing different events that can occur
//! during tests execution process.

use super::types::TestDesc;
use super::test_result::TestResult;
use super::time::TestExecTime;

#[derive(Clone)]
pub enum TestEvent {
    TeFiltered(Vec<TestDesc>),
    TeWait(TestDesc),
    TeResult(TestDesc, TestResult, Option<TestExecTime>, Vec<u8>),
    TeTimeout(TestDesc),
    TeFilteredOut(usize),
}

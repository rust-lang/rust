use std::rc::Rc;

use super::{data_race, weak_memory};
use crate::concurrency::GenmcCtx;
use crate::{VisitProvenance, VisitWith};

pub enum GlobalDataRaceHandler {
    /// No data race detection will be done.
    None,
    /// State required to run in GenMC mode.
    /// In this mode, the program will be executed repeatedly to explore different concurrent executions.
    /// The `GenmcCtx` must persist across multiple executions, so it is behind an `Rc`.
    ///
    /// The `GenmcCtx` has several methods with which to inform it about events like atomic memory accesses.
    /// In GenMC mode, some functionality is taken over by GenMC:
    /// - Memory Allocation:    Allocated addresses need to be consistent across executions, which Miri's allocator doesn't guarantee
    /// - Scheduling:           To influence which concurrent execution we will explore next, GenMC takes over scheduling
    /// - Atomic operations:    GenMC will ensure that we explore all possible values that the memory model allows
    ///   an atomic operation to see at any specific point of the program.
    Genmc(Rc<GenmcCtx>),
    /// The default data race detector for Miri using vector clocks.
    Vclocks(Box<data_race::GlobalState>),
}

#[derive(Debug)]
pub enum AllocDataRaceHandler {
    None,
    Genmc,
    /// Data race detection via the use of vector clocks.
    /// Weak memory emulation via the use of store buffers (if enabled).
    Vclocks(data_race::AllocState, Option<weak_memory::AllocState>),
}

impl GlobalDataRaceHandler {
    pub fn is_none(&self) -> bool {
        matches!(self, GlobalDataRaceHandler::None)
    }

    pub fn as_vclocks_ref(&self) -> Option<&data_race::GlobalState> {
        if let Self::Vclocks(data_race) = self { Some(data_race) } else { None }
    }

    pub fn as_vclocks_mut(&mut self) -> Option<&mut data_race::GlobalState> {
        if let Self::Vclocks(data_race) = self { Some(data_race) } else { None }
    }

    pub fn as_genmc_ref(&self) -> Option<&GenmcCtx> {
        if let Self::Genmc(genmc_ctx) = self { Some(genmc_ctx) } else { None }
    }
}

impl AllocDataRaceHandler {
    pub fn as_vclocks_ref(&self) -> Option<&data_race::AllocState> {
        if let Self::Vclocks(data_race, _weak_memory) = self { Some(data_race) } else { None }
    }

    pub fn as_vclocks_mut(&mut self) -> Option<&mut data_race::AllocState> {
        if let Self::Vclocks(data_race, _weak_memory) = self { Some(data_race) } else { None }
    }

    pub fn as_weak_memory_ref(&self) -> Option<&weak_memory::AllocState> {
        if let Self::Vclocks(_data_race, weak_memory) = self { weak_memory.as_ref() } else { None }
    }

    pub fn as_weak_memory_mut(&mut self) -> Option<&mut weak_memory::AllocState> {
        if let Self::Vclocks(_data_race, weak_memory) = self { weak_memory.as_mut() } else { None }
    }
}

impl VisitProvenance for GlobalDataRaceHandler {
    fn visit_provenance(&self, visit: &mut VisitWith<'_>) {
        match self {
            GlobalDataRaceHandler::None => {}
            GlobalDataRaceHandler::Vclocks(data_race) => data_race.visit_provenance(visit),
            GlobalDataRaceHandler::Genmc(genmc_ctx) => genmc_ctx.visit_provenance(visit),
        }
    }
}

impl VisitProvenance for AllocDataRaceHandler {
    fn visit_provenance(&self, visit: &mut VisitWith<'_>) {
        match self {
            AllocDataRaceHandler::None => {}
            AllocDataRaceHandler::Genmc => {}
            AllocDataRaceHandler::Vclocks(data_race, weak_memory) => {
                data_race.visit_provenance(visit);
                weak_memory.visit_provenance(visit);
            }
        }
    }
}

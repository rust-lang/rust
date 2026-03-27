# BCI-to-OpenClaw Integration Prototype -- Task Breakdown (v2)

## Dependency Graph

```
Task 1 (Scaffold)
  |
  +---> Task 2 (DSP + Classifier)
  |       |
  |       +---> Task 4 (State Server) [also depends on Task 1]
  |       |       |
  |       |       +---> Task 5 (Integration: Reader+DSP+Server)
  |       |
  +---> Task 3 (BrainFlow Reader)
  |       |
  |       +---> Task 5 (Integration)
  |
  +---> Task 6 (OpenClaw Plugin) [depends on Task 1, Task 4]
  |
  Task 7 (SKILL.md) -- no deps
  |
  Tasks 5,6,7 ---> Task 8 (End-to-End Test)
```

---

## Task 1: Project Scaffolding
**Branch:** `feature/project-scaffold`
**Depends on:** None
**Estimate:** 1h
**Status:** DONE

## Task 2: DSP Module + Classifier
**Branch:** `feature/dsp-classifier`
**Depends on:** Task 1
**Estimate:** 3h
**Status:** DONE (dsp.py, classifier.py, test_dsp.py, test_classifier.py)

## Task 3: BrainFlow Reader
**Branch:** `feature/brainflow-reader`
**Depends on:** Task 1
**Estimate:** 2h
**Status:** DONE (brainflow_reader.py)

## Task 4: State Server (FastAPI)
**Branch:** `feature/state-server`
**Depends on:** Task 1, Task 2
**Estimate:** 2h
**Status:** DONE (server.py, state_manager.py)

## Task 5: Integration (Reader + DSP + Server)
**Branch:** `feature/integration`
**Depends on:** Task 2, Task 3, Task 4
**Estimate:** 2h
**Status:** DONE (__main__.py, config.py)

## Task 6: OpenClaw Plugin
**Branch:** `feature/openclaw-plugin`
**Depends on:** Task 1, Task 4
**Estimate:** 2h
**Status:** DONE (index.ts, tests)

## Task 7: BCI SKILL.md
**Branch:** `feature/bci-skill`
**Depends on:** None
**Estimate:** 1h
**Status:** DONE (skill/SKILL.md)

## Task 8: End-to-End Test
**Branch:** `feature/e2e-tests`
**Depends on:** Task 5, Task 6, Task 7
**Estimate:** 2h
**Status:** TODO

**Description:** Start signal processor in synthetic mode, verify GET /state returns valid JSON conforming to schemas, mock plugin fetch, verify natural_language_summary injection.

**Acceptance criteria:**
- [ ] Signal processor starts with synthetic board and serves on port 7680
- [ ] GET /state returns valid state_response with available=true
- [ ] GET /health returns valid health_response with status="ok"
- [ ] BCIState fields conform to bci_state.schema.json
- [ ] Plugin buildContext extracts summary correctly
- [ ] Plugin handles server-down and stale-data gracefully
- [ ] End-to-end latency (server response) < 50ms

---

## Summary

| Task | Estimate | Status |
|------|----------|--------|
| 1. Scaffold | 1h | DONE |
| 2. DSP + Classifier | 3h | DONE |
| 3. BrainFlow Reader | 2h | DONE |
| 4. State Server | 2h | DONE |
| 5. Integration | 2h | DONE |
| 6. OpenClaw Plugin | 2h | DONE |
| 7. SKILL.md | 1h | DONE |
| 8. E2E Test | 2h | TODO |
| **Total** | **~15h** | **7/8 done** |

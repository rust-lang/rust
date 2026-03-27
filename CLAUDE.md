# CLAUDE.md - Project Instructions

## Mandatory Development Workflow

Every request to modify code MUST follow this exact workflow. Steps 1-6 are strictly sequential (do NOT parallelize). Steps 7-8 are parallel. Steps 9-11 are sequential.

### Phase 1: Research (Sequential)

**Step 1 - Research Agent**
Spawn a research agent to investigate the topic, codebase, APIs, libraries, and any relevant context. Return comprehensive findings.

**Step 2 - Research Audit Agent**
Spawn an audit agent to review the research findings for accuracy, completeness, gaps, and incorrect assumptions. Flag anything missing or wrong.

### Phase 2: Design (Sequential)

**Step 3 - Schema Generator / High-Level Changes**
Based on audited research, spawn an agent to generate schemas, data models, and identify the high-level changes needed. Define interfaces and contracts.

**Step 4 - Schema Audit Agent**
Spawn an audit agent to verify schemas for: correctness, consistency, security, real-world validity, KISS compliance, and cross-schema compatibility.

**Step 5 - System Architecture Agent**
Spawn an agent to design the system architecture and implementation plan based on the audited schemas. Follow KISS, Unix philosophy, separation of concerns. Define components, data flow, technology choices, and error handling.

**Step 6 - Architecture Audit Agent**
Spawn an audit agent to review the system architecture for: feasibility, simplicity, missing concerns, over-engineering, and alignment with the audited schemas and research.

### Phase 3: Implementation (Parallel)

**Step 7 - Implementation Agents (PARALLEL)**
Spawn multiple implementation agents in parallel, one per component/task. Each agent implements its assigned piece according to the audited architecture.

**Step 8 - Implementation Audit Agents (PARALLEL)**
For each implementation agent's changes, spawn a corresponding audit agent to review: code quality, schema compliance, test coverage, security, and adherence to the architecture plan.

### Phase 4: Integration & PR (Sequential)

**Step 9 - System Architecture Review**
Spawn an agent to review how all implemented pieces fit together. Verify the system architecture plan maps correctly to the implementation details. Flag integration issues.

**Step 10 - PR Agent**
Create the pull request with a clear summary, test plan, and all changes.

**Step 11 - PR Review Agent**
Spawn a review agent to perform a final code review of the entire PR: correctness, style, tests, security, documentation, and completeness.

---

## General Principles

- KISS (Keep It Simple, Stupid)
- Unix philosophy: do one thing well, compose small tools
- Make it work, then make it right, then make it fast
- Fail fast, fail loudly
- Prefer boring technology
- Don't add features beyond what was asked
- Measure twice, cut once

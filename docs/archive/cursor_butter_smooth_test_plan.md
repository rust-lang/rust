# Cursor "Butter Smooth" Test Plan

## Overview
This document describes the test plan to verify that cursor motion is "butter smooth" via late-latched overlay composition.

## Test Prerequisites
- Build the bloom compositor with cursor metrics enabled
- Run on QEMU or real hardware with display support
- Have a cursor asset loaded (or use the crosshair fallback)

## Test 1: Cursor Wiggle Test
**Objective**: Verify cursor moves smoothly and damage is minimal

**Steps**:
1. Launch bloom compositor
2. Move the cursor in circles continuously for ~10 seconds
3. Observe cursor metrics in logs (printed every 120 frames)

**Expected Results**:
- `cursor_moves` should increment with each movement event
- `cursor_rasterizations` should be 0 or 1 (only on first render)
- `damage_rects_from_cursor` should be approximately `2 * cursor_moves` (old + new position)
- `frames_cursor_only` should be > 0 if no other UI activity
- No cursor trails visible on screen
- Cursor motion feels smooth, no stutter or lag

**Success Criteria**:
```
[cursor metrics] frame=120 moves=45 rasterizations=1 cursor_only_frames=42 damage_rects=90
```
- Rasterizations ≤ 1 during movement phase
- Damage rects ≈ 2 per move
- Cursor only frames > 0 when UI is idle

## Test 2: Cursor Under Heavy Load
**Objective**: Verify cursor remains smooth even when UI is busy

**Steps**:
1. Launch bloom compositor
2. Open a window with busy repaint (e.g., animation, scrolling text)
3. Move the cursor while the window is repainting
4. Check cursor metrics

**Expected Results**:
- Cursor motion remains smooth despite heavy window repaints
- `cursor_rasterizations` still 0 during movement
- `cursor_only_frames` may be 0 (expected if windows are also damaged)
- No visible stutter in cursor motion
- Cursor updates feel decoupled from window redraws

**Success Criteria**:
- Visual: Cursor moves smoothly, no frame skips
- Metrics: `cursor_rasterizations` = 0 during movement phase

## Test 3: Cursor Shape Change
**Objective**: Verify cursor rasterization only happens on shape change

**Steps**:
1. Launch bloom compositor
2. Start with default cursor
3. Change cursor asset/shape
4. Move cursor after shape change

**Expected Results**:
- One rasterization when shape changes
- Zero rasterizations during subsequent movement
- New cursor shape displays correctly
- No cursor trails with new shape

**Success Criteria**:
```
[cursor_rasterizer] rasterizing cursor snapshot gen=2
[cursor metrics] frame=240 moves=67 rasterizations=2 ...
```
- Exactly 1 new rasterization after shape change
- Movement after shape change: 0 rasterizations

## Test 4: Cursor Damage Isolation
**Objective**: Verify cursor damage never triggers full-frame redraws

**Steps**:
1. Launch bloom compositor with debug damage visualization enabled
2. Move cursor while monitoring damage rectangles
3. Check logs for damage causes

**Expected Results**:
- Cursor movement creates exactly 2 damage rects (old + new position)
- Damage rects are small (~35x35 pixels + 2px margin)
- No full-frame damage from cursor movement alone
- Damage cause is `DamageCause::CursorMoved`

**Success Criteria**:
- Damage rects visible as small green boxes (cursor damage color)
- No red full-frame damage overlays from cursor movement
- Damage rect count ≤ 2 per cursor move when UI is idle

## Test 5: No Cursor Trails
**Objective**: Verify old cursor position is properly erased

**Steps**:
1. Launch bloom compositor
2. Move cursor across the screen in various patterns
3. Visually inspect for cursor trails

**Expected Results**:
- No visible cursor trails
- Old cursor position is fully erased
- Only one cursor visible at any time

**Success Criteria**:
- Visual inspection: No cursor trails
- Old cursor rect properly damaged and redrawn

## Metrics Interpretation

### Good Metrics (Butter Smooth)
```
[cursor metrics] frame=120 moves=50 rasterizations=1 cursor_only_frames=48 damage_rects=100
```
- High cursor_only_frames (cursor-only updates)
- Low rasterizations (1 initial, 0 during movement)
- damage_rects ≈ 2 * moves

### Bad Metrics (Not Smooth)
```
[cursor metrics] frame=120 moves=50 rasterizations=50 cursor_only_frames=0 damage_rects=500
```
- High rasterizations (re-rasterizing on every move)
- No cursor_only_frames (always full recompose)
- Excessive damage_rects

## Automated Test (Future)
Consider adding an automated test that:
1. Injects synthetic mouse move events
2. Counts rasterizations and damage rects
3. Asserts rasterizations = 1 and damage_rects ≈ 2 * moves

## Notes
- Cursor metrics are logged every 120 frames (~2 seconds at 60 FPS)
- The crosshair fallback (when no cursor asset) should behave the same way
- Cursor-only fast path is a future optimization (currently documented as TODO)

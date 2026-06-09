use crate::{CONFIG_CHANGE_HISTORY, find_recent_config_change_ids};

#[test]
fn test_find_recent_config_change_ids() {
    // If change-id is greater than the most recent one, result should be empty.
    assert!(find_recent_config_change_ids(usize::MAX).is_empty());

    // There is no change-id equal to or less than 0, result should include the entire change history.
    assert_eq!(find_recent_config_change_ids(0).len(), CONFIG_CHANGE_HISTORY.len());
}

pub enum PublishedFileVisibility {
    Public = sys::ERemoteStoragePublishedFileVisibility_k_ERemoteStoragePublishedFileVisibilityPublic as i32,
    FriendsOnly = sys::ERemoteStoragePublishedFileVisibility_k_ERemoteStoragePublishedFileVisibilityFriendsOnly as i32,
    Private = sys::ERemoteStoragePublishedFileVisibility_k_ERemoteStoragePublishedFileVisibilityPrivate as i32,
}
